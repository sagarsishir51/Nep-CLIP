import torch.nn as nn
from transformers import AutoModel

class NepClipEncoder(nn.Module):
  def __init__(self, student_id):
      super().__init__()
      self.student = AutoModel.from_pretrained(student_id)
      self.student.eval()
      self.projection = nn.Linear(768, 512)

  def get_nepali_text_embeddings(self, student_inputs):
    text_embeddings = self.student(**student_inputs)
    return text_embeddings.pooler_output

  def forward(self,student_inputs):
    nepali_embeddings = self.get_nepali_text_embeddings(student_inputs=student_inputs)
    student_projected_embeddings = self.projection(nepali_embeddings)
    return  student_projected_embeddings