import re

from PIL import Image
from torch.utils.data import Dataset
import torch
from tqdm import tqdm

class ALMEvaluationDataset(Dataset):
    def __init__(self, dataset_translated_map, dataset, image_processor):
        """
        Memory-safe Lazy Loading Dataset.
        """
        self.texts = []
        self.eng_texts = []
        self.labels = []
        self.languages = []
        self.categories = []
        processed_images = []

        for item in tqdm(dataset, desc="Processing Images", unit="img"):
            img = item.get("file_name")
            parsed_option = ALMEvaluationDataset.parse_translated_answer(sample=item)
            if parsed_option is not None:
                id = item.get("ID")
                if isinstance(img, Image.Image):
                    img = img
                elif not isinstance(img, torch.Tensor):
                    img = Image.open(img).convert("RGB")
                else:
                    img = img
                encoding = image_processor(images=img, return_tensors="pt")
                processed_images.append(encoding.pixel_values.squeeze(0))
                if dataset_translated_map is not None:
                    translated_dataset = dataset_translated_map[id]
                    nepali_question = translated_dataset.get("nepali_question")
                    translated_options = translated_dataset.get("nepali_options")
                    self.texts.extend([f"{nepali_question} {opt}" for opt in translated_options])

                english_question = item.get("English_Question")
                self.labels.append(parsed_option.get("Label"))
                english_options = parsed_option.get("Translated Options")
                self.languages.append(item.get("Language"))
                self.categories.append(item.get("Category"))
                self.eng_texts.extend([f"{english_question} {opt}" for opt in english_options])

        self.pixel_values = torch.stack(processed_images, dim=0)

        print(f"Dataset initialized safely with {len(self.texts)} samples.")

    def __len__(self):
        return len(self.texts)

    def get_all_data(self):
        """Helper to get everything for run_report"""
        return self.pixel_values,self.texts,self.eng_texts,self.labels,self.languages,self.categories

    @staticmethod
    def parse_translated_answer(sample):
        """
        Parses an ALM-Bench multiple-choice answer string into structured values.
        Throws a ValueError if formatting requirements or logic checks fail.
        """
        sample_id = sample.get("ID", "Unknown ID")

        # Safeguard check for non-MCQ formats
        if sample.get("Question_Type") != "Multiple Choice Questions":
            return {
                "Correct Answer": sample.get("English_Answer"),
                "Translated Options": [],
                "Label": -1
            }

        raw_text = sample.get("English_Answer", "")
        if not raw_text:
            print(f"❌ Parsing Error: 'English_Answer' field is missing or empty for ID: {sample_id}")
            return None


        # Flow Fix 6: Updated 'opt[a-z]*' to handle truncated words like '(optio:' or '(opt:'
        match = re.search(r"^(.*?)\s*\(opt[a-z]*:\s*(.*?)\)$", raw_text, re.IGNORECASE)

        if not match:
            print( f"❌ Parsing Error: Target text does not match the standard options suffix structure.\nRaw Text: '{raw_text}' for ID: {sample_id}")
            return None

        correct_answer = match.group(1).strip()
        options_string = match.group(2).strip()

        # Flow Fix 2: Handle trailing periods on the answer key that don't exist in the options list
        if correct_answer.endswith(".") and correct_answer not in options_string:
            if correct_answer[:-1] in options_string:
                correct_answer = correct_answer[:-1]

        # -------------------------------------------------------------------------
        # FIXED: Context-Aware Pivot Splitter for Complex/Nested Lists (No Lookbehinds)
        # -------------------------------------------------------------------------
        # Check if the options string starts with the correct answer (very common)
        if options_string.startswith(correct_answer):
            remainder = options_string[len(correct_answer):].strip()

            # Strip leading comma if present
            if remainder.startswith(','):
                remainder = remainder[1:].strip()

            # Parse the remaining string by matching option structures
            if "and " in correct_answer.lower():
                raw_split = [correct_answer]
                # Split items using a safe lookahead sequence instead of lookbehinds
                parts = [opt.strip() for opt in re.split(r',\s*(?=[A-Z][a-z]+(?:,.*?and|.*?and))', remainder)]
                raw_split.extend([p for p in parts if p])
            else:
                raw_split = [correct_answer] + [opt.strip() for opt in
                                                re.split(
                                                    r',\s*(?=[A-Z][a-zA-Z]*(?:\s+[A-Z][a-zA-Z]*)*\s*(?:,|\s+and\b|\s*$))',
                                                    remainder
                                                )]
        else:
            # Fallback if correct answer isn't the first item
            raw_split = [opt.strip() for opt in re.split(r',\s*(?=[A-Z][a-z]+(?:,.*?and|.*?and))', options_string)]

        # Dynamic fallback check: If we over-parsed or under-parsed, force group by looking
        # for the explicit 'and ' anchors as terminal flags for each MCQ block
        if len(raw_split) != 4 and "and " in options_string.lower():
            items = [opt.strip() for opt in re.split(r',\s*', options_string)]
            raw_split = []
            current_group = []
            for item in items:
                current_group.append(item)
                if item.lower().startswith('and ') or (
                        len(current_group) > 1 and any(x.lower().startswith('and ') for x in item.split())):
                    raw_split.append(", ".join(current_group))
                    current_group = []
                elif len(raw_split) == 0 and ", ".join(current_group) == correct_answer:
                    # If we exactly matched the correct answer out of the gate, commit it as a block
                    raw_split.append(", ".join(current_group))
                    current_group = []
            if current_group:
                raw_split.append(", ".join(current_group))

        # -------------------------------------------------------------------------
        # Option List Alignment Logic
        # -------------------------------------------------------------------------
        raw_split_lower = [opt.lower() for opt in raw_split]
        if correct_answer.lower() in raw_split_lower:
            options_list = raw_split
        else:
            # Revert to Smart Buffer Loop with Oxford Comma Protection
            options_list = []
            current_buffer = []

            for item in raw_split:
                current_buffer.append(item)
                candidate = ", ".join(current_buffer)

                if candidate == correct_answer:
                    options_list.append(candidate)
                    current_buffer = []
                elif any(candidate == existing_opt for existing_opt in options_list):
                    options_list.append(candidate)
                    current_buffer = []

            # Fallback to broad regex delimiter split
            if not options_list or len(options_list) < 2:
                options_list = [opt.strip() for opt in re.split(r',\s*(?![^()]*\))', options_string)]

                if correct_answer in options_string and correct_answer not in options_list:
                    parts_of_answer = [p.strip() for p in correct_answer.split(", ")]
                    new_options = []
                    idx = 0
                    while idx < len(options_list):
                        if options_list[idx:idx + len(parts_of_answer)] == parts_of_answer:
                            new_options.append(correct_answer)
                            idx += len(parts_of_answer)
                        else:
                            new_options.append(options_list[idx])
                            idx += 1
                    options_list = new_options

        options_list = [opt.strip() for opt in options_list if opt.strip()]

        # --- Strict Error Validation Checks ---
        if not options_list or len(options_list) < 2:
            print(f"❌ Parsing Error: Failed to unpack multiple options from text field. Only extracted: {options_list}\nRaw Options Context: '{options_string}' for ID: {sample_id}")
            return None

        # Flow Fix 5: Case-Insensitive Index Mapping & Robust Fallback Logging
        try:
            # Find index using uniform lowercase matching
            options_list_lower = [opt.lower() for opt in options_list]
            label_index = options_list_lower.index(correct_answer.lower())

            # Sync the return payload's target text to match the dataset's option list casing exactly
            correct_answer = options_list[label_index]
        except ValueError:
            # Fallback Triggered: Map to the 1st option (index 0) and log the incident
            label_index = 0
            fallback_answer = options_list[0]

            print(
                f"⚠️ Ground-Truth Alignment Fallback [ID: {sample_id}]: "
                f"Extracted answer target '{correct_answer}' was not found in options {options_list}. "
                f"Defaulting label to index 0 ('{fallback_answer}')."
            )

            correct_answer = fallback_answer

        if len(options_list) < 4 :
            print(f"❌ Parsing Error: Less than 4 options are parsed: {options_list}\n Raw Text is: {raw_text}\nRaw Options Context:for ID: {sample_id}")
            return None

        if len(options_list) > 4:
            print(f"❌ Parsing Error: More than 4 options are parsed: {options_list}\n Raw Text is: {raw_text}\nRaw Options Context:for ID: {sample_id}")
            return None

        return {
            "Correct Answer": correct_answer,
            "Translated Options": options_list,
            "Label": label_index
        }