from sentinelhub import (
    CRS,
    BBox,
    bbox_to_dimensions,
    SHConfig,
)
from dotenv import load_dotenv
import os
# Save the image data
import matplotlib.pyplot as plt
from sentinel_hub_image_downloader import SentinelHubImageDownloader
import imageio

config = SHConfig()

load_dotenv('.env.local') 

config.sh_client_id = os.getenv("SH_CLIENT_ID")
config.sh_client_secret = os.getenv("SH_CLIENT_SECRET")

if not config.sh_client_id or not config.sh_client_secret:
    print("Warning! To use Process API, please provide the credentials (OAuth client ID and client secret).")


coordinates = (13.294333, 52.454927, 13.500205, 52.574409)


resolution = 10
bbox = BBox(bbox=coordinates, crs=CRS.WGS84)
size = bbox_to_dimensions(bbox, resolution=resolution)

print(f"Image shape at {resolution} m resolution: {size} pixels")



evalscript_rgb = """
//VERSION=3
function setup() {
  return {
    input: ["B04","B03","B02", "dataMask"],
    output: { bands: 4 }
  };
}

function evaluatePixel(sample) {
  
  return [2.5 * sample.B04, 2.5 * sample.B03, 2.5 * sample.B02, sample.dataMask];
}
"""

evalscript_irb = """
//VERSION=3
function setup() {
  return {
    input: ["B08","B04","B02", "dataMask"],
    output: { bands: 4 }
  };
}

function evaluatePixel(sample) {
  
  return [2.5 * sample.B08, 2.5 * sample.B04, 3.5 * sample.B02, sample.dataMask];
}
"""


evalscript_gray = """
//VERSION=3
function setup() {
  return {
    input: ["B02"],
    output: { bands: 1 }
  };
}

function evaluatePixel(sample) {
  return [ 3.5 * sample.B02];
}
"""

downloader = SentinelHubImageDownloader(bbox, size, config, max_cloud_coverage=5)

# downloader.set_evalscript(evalscript_rgb)
downloader.set_time_interval(("2024-04-17", "2024-05-17"))
# downloader.save_image('output_image_rgb.png')

# downloader.set_evalscript(evalscript_irb)
# downloader.save_image('output_image_irb.png')


downloader.set_evalscript(evalscript_gray)
downloader.set_time_interval(("2024-04-17", "2024-05-17"))

# Download the image data
image_data = downloader.download_image()[0]


# plt.imsave('output_image_blue.png', image_data, cmap='gray')

imageio.imsave('output_image_blue3.png', image_data)

# downloader.save_image('output_image_gray.png')
