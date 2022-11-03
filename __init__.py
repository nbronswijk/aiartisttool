import logging
import base64
from io import BytesIO
import azure.functions as func

from final_function import NeuralStyleTransferV2 as NST

def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')

    input_picture = req.params.get('input_picture')
#    if not input_picture:
#        try:
#            req_body = req.get_json()
#        except ValueError:
#            pass
#        else:
#            input_picture = req_body.get('input_picture')

    style = req.params.get('style')
#    if not style:
#        try:
#            req_body = req.get_json()
#        except ValueError:
#            pass
#        else:
#            style = req_body.get('style')   


    if (input_picture and style):
        NST = NST(input_picture=input_picture, style_choice = style)
        buffered = BytesIO()
        NST.artwork.save(buffered,format="JPEG")
        output_data = base64.b64encode(buffered.getvalue())
        return func.HttpResponse(output_data)
    else:
        return func.HttpResponse(
             "We didn't receive all the necessary information",
             status_code=200
        )