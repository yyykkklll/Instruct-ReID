import json
import os
import asyncio
import base64
import logging
from pathlib import Path
from tqdm import tqdm
from PIL import Image, UnidentifiedImageError
import io
import aiofiles
from openai import AsyncOpenAI

# 设置日志
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# API 配置
API_KEY = "sk-A7ZqMwJfFt4tIPDWgjOXjx2OmMPUWweDXELkjkjREBrD4dgn"
API_URL = "https://www.chataiapi.com/v1"
client = AsyncOpenAI(api_key=API_KEY, base_url=API_URL)

# 数据集路径
DATASET_ROOT = "Image/CUHK-PEDES"
ANNOTATION_PATH = os.path.join(DATASET_ROOT, "annotations/caption_all.json")
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 输出文件路径
CAPTION_ID_PATH = os.path.join(OUTPUT_DIR, "caption_id.json")
CAPTION_CLOTH_PATH = os.path.join(OUTPUT_DIR, "caption_cloth.json")
TEMP_ID_PATH = os.path.join(OUTPUT_DIR, "caption_id_temp.json")
TEMP_CLOTH_PATH = os.path.join(OUTPUT_DIR, "caption_cloth_temp.json")

# Prompt 设计
IDENTITY_PROMPT = """
You are an expert in person re-identification. Given an image of a person, generate a concise description of their identity-related characteristics, including gender, body type, posture, orientation, and any notable actions. The description should be clear, specific, and no longer than 50 words. Avoid describing clothing or accessories unless they are directly relevant to actions (e.g., holding a bag). Output the description as a single string.
Example input image: A woman facing left, head down, heavy set, carrying something in her hand.
Example output: "Woman, heavy set, facing left, head down, carrying something."

Now, analyze the provided image and generate the identity description.
"""

CLOTH_PROMPT = """
You are an expert in person re-identification. Given an image of a person, generate a concise description of their clothing, including specific clothing types (e.g., T-shirt, jeans, dress), colors, sleeve/length details (e.g., short-sleeve, ankle-length), and notable patterns or features (e.g., stripes, buttons, rips). The description should be clear, specific, and no longer than 50 words. Avoid describing non-clothing elements like gender or posture. Output the description as a single string.
Example input image: A person in a navy blue T-shirt, black shorts, and no patterns.
Example output: "Navy blue short-sleeve T-shirt, knee-length black shorts, no patterns."
Example input image: A person in a white blouse and blue jeans with rips.
Example output: "White cold-shoulder blouse, ankle-length blue jeans with ripped details."
Example input image: A person in a light blue shirt and black trousers.
Example output: "Light blue long-sleeve button-up shirt, black tailored trousers."

Now, analyze the provided image and generate the clothing description.
"""

def encode_image(image_path):
    """将图像编码为 Base64 格式，必要时转换为 JPEG"""
    try:
        ext = os.path.splitext(image_path)[1].lower()
        if ext not in ['.jpg', '.jpeg', '.bmp', '.png']:
            logger.error(f"Unsupported image format: {image_path}")
            return None, None

        with Image.open(image_path) as img:
            img = img.convert('RGB')  # 转换为 RGB
            buffer = io.BytesIO()
            img.save(buffer, format='JPEG', quality=85)  # 统一转 JPEG
            image_data = buffer.getvalue()

        encoded = base64.b64encode(image_data).decode("utf-8")
        mime_type = "image/jpeg"
        return encoded, mime_type
    except UnidentifiedImageError:
        logger.error(f"Invalid image file: {image_path}")
        return None, None
    except Exception as e:
        logger.error(f"Failed to encode image {image_path}: {e}")
        return None, None

async def generate_description(image_path, prompt, retries=3, timeout=30, quota_wait=60):
    """异步调用 OpenAI API 生成描述，自动处理配额限制"""
    base64_image, image_mime_type = encode_image(image_path)
    if not base64_image:
        return None

    for attempt in range(retries):
        try:
            response = await client.chat.completions.create(
                model="gemini-2.0-flash",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": f"data:{image_mime_type};base64,{base64_image}"}},
                        ],
                    }
                ],
                max_tokens=100,
                timeout=timeout,
            )
            description = response.choices[0].message.content.strip()
            return description
        except Exception as e:
            if "rate_limit_exceeded" in str(e).lower() or "429" in str(e):
                logger.warning(f"API quota exceeded for {image_path}. Waiting {quota_wait} seconds before retrying...")
                await asyncio.sleep(quota_wait)  # 等待配额恢复
                continue
            logger.error(f"Failed to generate description for {image_path}: {e}")
            if attempt < retries - 1:
                logger.warning(f"Retrying after 10 seconds...")
                await asyncio.sleep(10)
    logger.error(f"Failed to generate description for {image_path} after {retries} attempts")
    return None

async def load_json_safe(file_path):
    """安全加载JSON文件，处理损坏或空文件"""
    try:
        async with aiofiles.open(file_path, "r") as f:
            content = await f.read()
            if not content.strip():
                logger.warning(f"Empty JSON file: {file_path}")
                return []
            return json.loads(content)
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in {file_path}: {e}. Starting with empty data.")
        return []
    except Exception as e:
        logger.error(f"Failed to load {file_path}: {e}")
        return []

file_lock = asyncio.Lock()  # 文件写入锁

class ConcurrencyController:
    """并发控制器，用于动态调整并发度"""
    def __init__(self, initial_concurrency=10):
        self.max_concurrency = initial_concurrency
        self.semaphore = asyncio.Semaphore(initial_concurrency)
        
    def reduce_concurrency(self):
        """降低并发度"""
        if self.max_concurrency > 1:
            self.max_concurrency = max(1, self.max_concurrency - 1)
            self.semaphore = asyncio.Semaphore(self.max_concurrency)
            logger.warning(f"Reduced concurrency to {self.max_concurrency}")

async def process_image(item, caption_id_data, caption_cloth_data, controller):
    """处理单张图像，动态调整并发"""
    async with controller.semaphore:
        id_ = item["id"]
        file_path = os.path.join(DATASET_ROOT, "imgs", item["file_path"])
        
        if not os.path.exists(file_path):
            logger.warning(f"Image not found: {file_path}")
            return

        # 生成身份描述
        identity_desc = None
        for attempt in range(3):
            identity_desc = await generate_description(file_path, IDENTITY_PROMPT)
            if identity_desc:
                break
            else:
                logger.warning(f"Quota issue for {file_path}, reducing concurrency and waiting...")
                controller.reduce_concurrency()
                await asyncio.sleep(60)  # 等待更长时间

        # 生成服装描述
        cloth_desc = None
        for attempt in range(3):
            cloth_desc = await generate_description(file_path, CLOTH_PROMPT)
            if cloth_desc:
                break
            else:
                logger.warning(f"Quota issue for {file_path}, reducing concurrency and waiting...")
                controller.reduce_concurrency()
                await asyncio.sleep(60)  # 等待更长时间

        # 更新结果集
        if identity_desc:
            caption_id_data.append({
                "id": id_,
                "file_path": item["file_path"],
                "identity": identity_desc
            })
        if cloth_desc:
            caption_cloth_data.append({
                "id": id_,
                "file_path": item["file_path"],
                "cloth": cloth_desc
            })

        # 使用锁保护文件写入
        async with file_lock:
            try:
                async with aiofiles.open(TEMP_ID_PATH, "w") as f:
                    await f.write(json.dumps(caption_id_data, indent=2))
                async with aiofiles.open(TEMP_CLOTH_PATH, "w") as f:
                    await f.write(json.dumps(caption_cloth_data, indent=2))
            except Exception as e:
                logger.error(f"Failed to save temporary files: {e}")

async def main():
    # 读取 caption_all.json
    try:
        async with aiofiles.open(ANNOTATION_PATH, "r") as f:
            annotations = json.loads(await f.read())
    except Exception as e:
        logger.error(f"Failed to load {ANNOTATION_PATH}: {e}")
        return

    # 加载已处理数据（断点续传）
    caption_id_data = await load_json_safe(TEMP_ID_PATH)
    caption_cloth_data = await load_json_safe(TEMP_CLOTH_PATH)
    processed_ids = set(item["id"] for item in caption_id_data)
    processed_ids.intersection_update(set(item["id"] for item in caption_cloth_data))  # 修复交集操作

    # 过滤已处理图像
    remaining_annotations = [item for item in annotations if item["id"] not in processed_ids]
    total_images = len(remaining_annotations)

    if not remaining_annotations:
        logger.warning("All images already processed.")
        return

    # 初始化并发控制器
    controller = ConcurrencyController(initial_concurrency=10)
    tasks = []
    for item in remaining_annotations:
        task = asyncio.create_task(
            process_image(item, caption_id_data, caption_cloth_data, controller)
        )
        tasks.append(task)

    # 使用 tqdm 显示进度
    for f in tqdm(asyncio.as_completed(tasks), total=total_images, desc="Processing images", unit="image"):
        await f

    # 保存最终结果
    async with file_lock:
        try:
            async with aiofiles.open(CAPTION_ID_PATH, "w") as f:
                await f.write(json.dumps(caption_id_data, indent=2))
            async with aiofiles.open(CAPTION_CLOTH_PATH, "w") as f:
                await f.write(json.dumps(caption_cloth_data, indent=2))
        except Exception as e:
            logger.error(f"Failed to save JSON files: {e}")

if __name__ == "__main__":
    asyncio.run(main())