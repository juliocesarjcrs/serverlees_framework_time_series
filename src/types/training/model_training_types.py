from pydantic import BaseModel

class OptionsSavePlot(BaseModel):
    type_storage: str
    output_dir: str
    names_folder: dict
    title: str

class NamesFolder(BaseModel):
    model_name: str