from .simple_lama import SimpleLama
from PIL import Image
from inferencesh import BaseApp, BaseAppInput, BaseAppOutput, File
from io import BytesIO
from PIL import Image
import time
import base64
import urllib.request

def load_image_from_url_or_path(url_or_path: str) -> Image.Image:
    print(f"Loading image from URL or path: {url_or_path}")
    if url_or_path.startswith("http") or url_or_path.startswith("https"):
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        req = urllib.request.Request(url_or_path, headers=headers)
        return Image.open(BytesIO(urllib.request.urlopen(req).read()))
    else:
        return Image.open(url_or_path)
    
def image_to_base64(image: Image.Image) -> str:
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

class AppInput(BaseAppInput):
    image: File
    mask: File

class AppOutput(BaseAppOutput):
    image: File

class App(BaseApp):
    lama: SimpleLama | None = None
    async def setup(self):
        self.lama = SimpleLama()

    async def run(self, app_input: AppInput) -> AppOutput:
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
        image = load_image_from_url_or_path(app_input.image.path).convert("RGB")

        # Download and open mask
        mask = load_image_from_url_or_path(app_input.mask.path).convert("L")

        result = self.lama(image, mask)
        result_path = "/tmp/result.png"
        result.save(result_path)
        output = AppOutput(image=File(path=result_path))
        return output

    async def unload(self):
        self.lama = None