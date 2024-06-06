import openeo
from datetime import date
# from openeo.processes import lte


connection = openeo.connect("https://openeo.dataspace.copernicus.eu")
print("Authenticate with OIDC authentication")

connection.authenticate_oidc()


area_of_interest = {
    "type": "Polygon",
    "coordinates": [[
        [13.294333, 52.454927],
        [13.500205, 52.454927],
        [13.500205, 52.574409],
        [13.294333, 52.574409],
        [13.294333, 52.454927]
    ]]
}

start_date = date(2024, 1, 1)
end_date = date(2024, 1, 3)

collection = connection.load_collection(
    "SENTINEL2_L2A",
    spatial_extent={"west": 13.294333, "south": 52.454927, "east": 13.500205, "north": 52.574409},
    temporal_extent=(start_date, end_date),
    bands=["B04", "B03", "B02"],
    cloud_coverage=20
    # properties = {
    #     "eo:cloud_cover": lambda x: lte( x, 20)
    #     }
)
print("Filtering bands and masking clouds")


# collection = collection.filter_bands(["B04", "B03", "B02", "B08"])

# {
#   "type": "FeatureCollection",
#   "features": [
#     {
#       "type": "Feature",
#       "geometry": {
#         "type": "Polygon",
#         "coordinates": [
#           [
#             [13.294333, 52.454927],
#             [13.500205, 52.454927],
#             [13.500205, 52.574409],
#             [13.294333, 52.574409],
#             [13.294333, 52.454927]
#           ]
#         ]
#       },
#       "properties": {}
#     }
#   ]
# }




# # Select the bands for R, G, and B
# R_band = collection.band("B04")  # Red band
# G_band = collection.band("B03")  # Green band
# B_band = collection.band("B02")  # Blue band

# # Merge the bands into a single cube
# RG = R_band.merge_cubes(G_band)
# RGB = RG.merge_cubes(B_band)

# # Save the result
# RGB = RGB.save_result(format="PNG")

# # Create and start the job
# job = RGB.create_job()
# job.start_and_wait().download_results()

# RG = R_BAND.merge_cubes(G_BAND)
# RGB = RG.merge_cubes(B_BAND)
# # collection = collection.process(
# #     process_id="mask_scl_dilation",
# #     arguments={"data": collection, "scl_band": "SCL", "mask_values": [3, 8, 9, 10, 11]}
# # )

output_format = "PNG"
output_path = "data_acquisition/satellite_images4"


# RGB = RGB.save_result(format=output_format)

# Start the job
job = collection.create_job(out_format=output_format, out_band="bands")
job.start_and_wait().get_results().download_files(output_path)



# # Download the results
# try:
#     job.get_results().download_files(output_path)
# except Exception as e:
#     print(f"An error occurred while downloading the files: {str(e)}")




