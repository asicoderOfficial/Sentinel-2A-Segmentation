from sentinelhub import (
    DataCollection,
    MimeType,
    SentinelHubRequest,
)
import matplotlib.pyplot as plt


class SentinelHubImageDownloader:
    def __init__(self, bbox, size, config, max_cloud_coverage=20):
        self.bbox = bbox
        self.size = size
        self.config = config
        self.evalscript = None
        self.time_interval = None
        self.max_cloud_coverage = max_cloud_coverage

    def set_evalscript(self, evalscript):
        self.evalscript = evalscript

    def set_time_interval(self, time_interval):
        self.time_interval = time_interval

    def download_image(self):
        request = SentinelHubRequest(
            evalscript=self.evalscript,
            input_data=[
                SentinelHubRequest.input_data(
                    data_collection=DataCollection.SENTINEL2_L2A,
                    time_interval=self.time_interval,
                    maxcc=self.max_cloud_coverage / 100.0,
                    
                )
            ],
            responses=[SentinelHubRequest.output_response("default", MimeType.PNG)],
            bbox=self.bbox,
            size=self.size,
            config=self.config,
        )
        return request.get_data()

    def save_image(self, filename):
        image = self.download_image()[0]
        print(f"Image type: {image.dtype}")
        plt.imsave(filename, image)