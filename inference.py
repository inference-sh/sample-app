from .simple_lama import SimpleLama
from PIL import Image
from pydantic import BaseModel, ConfigDict
from io import BytesIO
from PIL import Image
from urllib.request import urlopen
import time
import base64

def load_image_from_url_or_path(url_or_path: str) -> Image.Image:
    print(f"Loading image from URL or path: {url_or_path}")
    if url_or_path.startswith("http") or url_or_path.startswith("https"):
        return Image.open(BytesIO(urlopen(url_or_path).read()))
    else:
        return Image.open(url_or_path)
    
def image_to_base64(image: Image.Image) -> str:
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

class BasePredictInput(BaseModel):
    pass

class BasePredictOutput(BaseModel):
    pass

class BasePredictor(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    async def setup(self):
        pass

    async def predict(self, input_data: BasePredictInput) -> BasePredictOutput:
        raise NotImplementedError("predict method must be implemented")

    async def unload(self):
        pass

class PredictInput(BasePredictInput):
    image: str  # PIL Image will be validated in predict method
    mask: str

class PredictOutput(BasePredictOutput):
    image: str

class Predictor(BasePredictor):
    lama: SimpleLama | None = None
    async def setup(self):
        self.lama = SimpleLama()

    async def predict(self, input_data: PredictInput) -> PredictOutput:
        # Download image and mask from URLs
        # Download and open image
        # Print a poem while loading for 10 seconds
        poem_lines = [
            "In circuits deep where data flows",
            "Through silicon and binary rows", 
            "The model learns with patient grace",
            "Each pixel finds its rightful place",
            "Through layers deep and neurons bright",
            "The patterns emerge into the light",
            "With gradients steep and loss declines",
            "The image heals through countless lines",
            "Until at last the task complete",
            "A masterpiece our eyes to greet"
        ]
        
        start_time = time.time()
        for line in poem_lines:
            if time.time() - start_time >= 10:
                break
            print(line)
            time.sleep(0.1)
        image = load_image_from_url_or_path(input_data.image).convert("RGB")

        # Download and open mask
        mask = load_image_from_url_or_path(input_data.mask).convert("L")

        result = self.lama(image, mask)
        result_path = "/tmp/result.png"
        result.save(result_path)
        output = PredictOutput(image=result_path)
        return output

    async def unload(self):
        self.lama = None