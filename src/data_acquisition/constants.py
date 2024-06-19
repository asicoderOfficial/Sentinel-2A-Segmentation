
EVALSCRIPT_V3_FORMAT = """
//VERSION=3
function setup() {{
    return {{
        input: {input_},
        output: {{ bands: {n_bands} }}
    }};
    }}
function evaluatePixel(sample) {{
    return [{expression}];
}}
"""

# RGB
RGB_INPUT_ = ["B04","B03","B02", "dataMask"]
RGB_N_BANDS = 4
RGB_EXPRESSION = "2.5 * sample.B04, 2.5 * sample.B03, 2.5 * sample.B02, sample.dataMask"
EVALSCRIPT_RGB = EVALSCRIPT_V3_FORMAT.format(input_=RGB_INPUT_, n_bands=RGB_N_BANDS, expression=RGB_EXPRESSION)

# IRB
IRB_INPUT_ = ["B08","B04","B02", "dataMask"]
IRB_N_BANDS = 4
IRB_EXPRESSION = "2.5 * sample.B08, 2.5 * sample.B04, 3.5 * sample.B02, sample.dataMask"
EVALSCRIPT_IRB = EVALSCRIPT_V3_FORMAT.format(input_=IRB_INPUT_, n_bands=IRB_N_BANDS, expression=IRB_EXPRESSION)

# GRAY
GRAY_INPUT_ = ["B02"]
GRAY_N_BANDS = 1
GRAY_EXPRESSION = "3.5 * sample.B02"
EVALSCRIPT_GRAY = EVALSCRIPT_V3_FORMAT.format(input_=GRAY_INPUT_, n_bands=GRAY_N_BANDS, expression=GRAY_EXPRESSION)

# All 13 bands
ALL_BANDS_INPUT_ = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12", "CLM"]
ALL_BANDS_N_BANDS = 13
ALL_BANDS_EXPRESSION = "sample.B01, sample.B02, sample.B03, sample.B04, sample.B05, sample.B06, sample.B07, sample.B08, sample.B8A, sample.B09, sample.B11, sample.B12, sample.CLM"
EVALSCRIPT_ALL_BANDS = EVALSCRIPT_V3_FORMAT.format(input_=ALL_BANDS_INPUT_, n_bands=ALL_BANDS_N_BANDS, expression=ALL_BANDS_EXPRESSION)

EVALSCRIPTS = {
    "RGB": EVALSCRIPT_RGB,
    "IRB": EVALSCRIPT_IRB,
    "GRAY": EVALSCRIPT_GRAY,
    "ALL": EVALSCRIPT_ALL_BANDS
}
