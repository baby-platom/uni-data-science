import asyncio
from pathlib import Path

import aiofiles
import aiohttp

from app.constants import SOURCE_DATASETS_DOWNLOAD_URL_MAPPING


async def download_file(
    http_session: aiohttp.ClientSession,
    *,
    target_file_path: Path,
    download_url: str,
    chunk_size: int = 8192,
) -> None:
    print(f"Downloading {target_file_path.name}")

    target_file_path.parent.mkdir(parents=True, exist_ok=True)

    async with http_session.get(download_url) as response:
        response.raise_for_status()

        async with aiofiles.open(target_file_path, "wb") as f:
            while True:
                chunk = await response.content.read(chunk_size)
                if not chunk:
                    break
                await f.write(chunk)

    print(f"Downloaded {target_file_path}")


async def main() -> None:
    tasks: list[asyncio.Task] = []

    async with aiohttp.ClientSession() as http_session:
        for (
            target_file_path,
            download_url,
        ) in SOURCE_DATASETS_DOWNLOAD_URL_MAPPING.items():
            task = asyncio.create_task(
                download_file(
                    http_session,
                    target_file_path=target_file_path,
                    download_url=download_url,
                )
            )
            tasks.append(task)

        await asyncio.gather(*tasks)


if __name__ == "__main__":
    asyncio.run(main())
