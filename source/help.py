from transformers import AutoProcessor

processor = AutoProcessor.from_pretrained(
    "microsoft/BiomedVLP-BioViL-T", 
    trust_remote_code=True
)

print(type(processor))  # Should now be BioVilViltProcessor