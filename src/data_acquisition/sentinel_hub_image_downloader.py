import re
from typing import Tuple, List
import logging

import numpy as np
from sentinelhub import (
    DataCollection,
    MimeType,
    SentinelHubRequest,
    SHConfig
)
import matplotlib.pyplot as plt


class SentinelHubImageDownloader:
    def __init__(self, bbox:Tuple[float, float, float, float], size:Tuple[float, float], config:SHConfig, max_cloud_coverage:float=0.0) -> None:
        """ Initializes the SentinelHubImageDownloader object

        Args:
            bbox (tuple): The bounding box of the area of interest, given as a tuple of 4 floats (min lon, min lat, max lon, max lat)
            size (tuple): The size of the image to download, given as a tuple of 2 integers (width, height)
            config (SHConfig): The configuration object for the SentinelHub API, containing the API credentials
            max_cloud_coverage (int, optional): The maximum allowable cloud coverage in percent. 
                Cloud coverage is a product average and not viewport accurate hence images may have more cloud cover than specified here. 
                Defaults to 0 (no clouds). Allowed range [0.0 - 100.0]
        
        Returns:
            None
        """        
        self.bbox = bbox
        self.size = size
        self.config = config
        self._evalscript = None
        self._time_interval = None
        self.max_cloud_coverage = max_cloud_coverage

    @property
    def evalscript(self) -> str:
        return self._evalscript

    @evalscript.setter
    def evalscript(self, value: str) -> None:
        if not isinstance(value, str):
            raise TypeError("evalscript must be a string")
        self._evalscript = value


    @property
    def time_interval(self) -> Tuple[str, str]:
        return self._time_interval

    @time_interval.setter
    def time_interval(self, value: Tuple[str, str]) -> None:
        if not (isinstance(value, tuple) and len(value) == 2 and all(isinstance(item, str) for item in value)):
            raise TypeError("time_interval must be a tuple of two strings")
        if not all(re.match(r"\d{4}-\d{2}-\d{2}", date) for date in value): raise ValueError("time_interval dates must be in the format YYYY-MM-DD")

        self._time_interval = value


    def download_image(self, extension:str='TIFF') -> List[np.ndarray]:
        """ Downloads the image data from the SentinelHub API, depending on the class properties.

        Args:
            extension (str, optional): The file extension of the image to download. Defaults to 'TIFF'. Allowed values are 'TIFF' and 'PNG'.
                Use PNG when downloading an image for visualization purposes (RGB bands). 
                Use TIFF when downloading an image for processing purposes (all 13 bands usually).

        Returns:
            List[np.ndarray]: A list of numpy arrays, each representing an image band.
        """        
        if extension == 'TIFF':
            mime_type = MimeType.TIFF
        elif extension == 'PNG':
            mime_type = MimeType.PNG
        else:
            raise ValueError("Only TIFF and PNG extensions are supported.")

        request = SentinelHubRequest(
            evalscript=self.evalscript,
            input_data=[
                SentinelHubRequest.input_data(
                    data_collection=DataCollection.SENTINEL2_L2A,
                    time_interval=self.time_interval,
                    maxcc=self.max_cloud_coverage / 100.0
                )
            ],
            responses=[SentinelHubRequest.output_response("default", mime_type)],
            bbox=self.bbox,
            size=self.size,
            config=self.config,
        )
        return request.get_data()
