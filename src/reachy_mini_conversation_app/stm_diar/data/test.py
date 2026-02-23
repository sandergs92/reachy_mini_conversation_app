from pathlib import Path

from nemo.collections.asr.models import SortformerEncLabelModel

ckpt = Path(__file__).parent / "diar_streaming_sortformer_4spk-v2.nemo"
model = SortformerEncLabelModel.restore_from(str(ckpt), map_location="cpu")
sm = model.sortformer_modules
print(f"chunk_left_context default = {sm.chunk_left_context}")
print(f"chunk_right_context default = {sm.chunk_right_context}")
print(f"chunk_len default = {sm.chunk_len}")
