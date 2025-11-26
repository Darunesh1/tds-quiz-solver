"""Test browser and file operations."""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.primitives.browser import browser_manager
from app.primitives.download import downloader
from app.primitives.parse import file_parser


async def test_browser():
    """Test browser automation."""
    print("\n" + "=" * 60)
    print("🌐 Testing Browser Manager")
    print("=" * 60)

    try:
        # Load demo page
        result = await browser_manager.load_page(
            "https://tds-llm-analysis.s-anand.net/demo"
        )

        print(f"✅ Page loaded: {result['url']}")
        print(f"   HTML length: {len(result['html'])} chars")
        print(f"   Text length: {len(result['text'])} chars")

        # Find submit URL
        submit_url = await browser_manager.find_submit_url(result["page"])
        print(f"   Submit URL: {submit_url or 'Not found'}")

        # Close page
        await result["page"].close()

    except Exception as e:
        print(f"❌ Browser test failed: {e}")
    finally:
        await browser_manager.close()


async def test_downloader():
    """Test file downloader."""
    print("\n" + "=" * 60)
    print("⬇️  Testing File Downloader")
    print("=" * 60)

    try:
        # Download a small test file
        test_url = "https://raw.githubusercontent.com/datasets/covid-19/main/data/countries-aggregated.csv"

        file_path = await downloader.download_file(test_url, "test-job")
        print(f"✅ Downloaded: {file_path}")
        print(f"   Size: {file_path.stat().st_size / 1024:.1f}KB")

        # Cleanup
        downloader.cleanup_job("test-job")
        print("✅ Cleanup completed")

    except Exception as e:
        print(f"❌ Download test failed: {e}")


def test_parser():
    """Test file parsers."""
    print("\n" + "=" * 60)
    print("📊 Testing File Parser")
    print("=" * 60)

    # Create test JSON file
    test_json = Path("/tmp/test.json")
    test_json.write_text('{"name": "test", "value": 123}')

    try:
        data = file_parser.parse_json(test_json)
        print(f"✅ JSON parsed: {data}")

    except Exception as e:
        print(f"❌ Parser test failed: {e}")
    finally:
        test_json.unlink(missing_ok=True)


async def main():
    print("\n🚀 Starting Batch 2 Tests\n")

    await test_browser()
    await test_downloader()
    test_parser()

    print("\n" + "=" * 60)
    print("✅ All Batch 2 tests completed!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
