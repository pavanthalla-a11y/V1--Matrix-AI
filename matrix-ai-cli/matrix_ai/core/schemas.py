from pydantic import BaseModel, EmailStr
from typing import Dict, Any, List, Optional

class DesignRequest(BaseModel):
    data_description: str
    num_records: int
    existing_metadata: Optional[Dict[str, Any]] = {}

class SynthesizeRequest(BaseModel):
    num_records: int
    metadata_dict: Dict[str, Any]
    seed_tables_dict: Dict[str, Any]
    user_email: EmailStr
    batch_size: Optional[int] = 1000
    use_fast_synthesizer: Optional[bool] = True

class StoreRequest(BaseModel):
    confirm_storage: bool
