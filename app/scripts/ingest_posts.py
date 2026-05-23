import os
import asyncio
import httpx
from dotenv import load_dotenv

load_dotenv()

AI_BASE = f"http://{os.getenv('HOST','127.0.0.1')}:{os.getenv('PORT','8001')}"

async def main():
    demo_posts = [
        {
            "postId": 101,
            "content": "Căn hộ 2PN tại Cầu Giấy, gần công viên. Giá ~3 tỷ. Nội thất cơ bản. Diện tích 70m2.",
            "metadata": {"city": "Hà Nội", "district": "Cầu Giấy", "type": "apartment", "summary_reason": "2PN, Cầu Giấy, khoảng 3 tỷ"},
        },
        {
            "postId": 102,
            "content": "Nhà phố 3 tầng tại Thanh Xuân, ngõ rộng. Giá 5.2 tỷ. Sổ đỏ đầy đủ.",
            "metadata": {"city": "Hà Nội", "district": "Thanh Xuân", "type": "house"},
        },
    ]

    async with httpx.AsyncClient(timeout=60) as client:
        r = await client.post(f"{AI_BASE}/api/ingest/posts", json={"posts": demo_posts, "replace": True})
        print("INGEST:", r.status_code, r.json())

        q = {
            "message": "Tìm căn hộ 2 phòng ngủ ở Cầu Giấy khoảng 3 tỷ",
        }
        r2 = await client.post(f"{AI_BASE}/api/chat", json=q)
        print("CHAT STATUS:", r2.status_code)
        print("CHAT RESPONSE:", r2.text[:500])
        
        # Chỉ parse JSON nếu status code là 200
        if r2.status_code == 200:
            print("CHAT JSON:", r2.json())
        else:
            print("CHAT ERROR - Full response:", r2.text)

if __name__ == "__main__":
    asyncio.run(main())
