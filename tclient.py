import tornado
import tornado.httpclient as httpclient
import pickle
from tornado import ioloop
import os
import sys
import cv2
import base64
import numpy as np
import csv
import datetime
from tensorflow.keras.applications import DenseNet121, ResNet50, ResNet101
from tensorflow.keras.models import Model
from vit_keras import vit   
import argparse
import logging
import tensorflow as tf
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
'''from tensorflow.keras import backend as K
K.clear_session()
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)
'''
# Configure logging to display INFO level messages and above
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')


imgs_path = os.listdir("Images")
parser = argparse.ArgumentParser(description="Im2Latex Training Program")
client_c = sys.argv[1]
sending_start_time = None
receiving_end_time = None
total_inference_time = None
metrics_headers = ['client_id','req_id', 'model_name', 'batch', 'split_index', 'time_batch','client_exec_time', 'server_exec_time', 'total_inference_time']
model_name = 'Resnet101'
split_index = 141

input_shape = (32, 32, 3)

models = dict()
models['DensetNet121'] = DenseNet121(weights='imagenet', include_top=False, input_shape=input_shape)
models['Resnet50'] = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
models['Resnet101'] = ResNet101(weights='imagenet', include_top=False, input_shape=input_shape)
models['VIT'] = vit.vit_b32(image_size=32,activation='softmax',pretrained=True,include_top=True,pretrained_top=False,classes=2)

'''
if model_name == 'VIT':
        split_indices = [3, 17]
    elif model_name == 'Resnet50'or model_name == 'Resnet101':
        split_indices = [5, 91]
    else:
        split_indices = [5, 94]
    split_index = split_indices[split_idx - 1]
'''
def get_model(model_name, split_index):
    if split_index != 0:
        layer = models[model_name].layers[split_index]
        #save splitted part of model belonging to the server in models dictionary
        return Model(inputs=models[model_name].input, outputs=layer.output)
    else:
        return models[model_name]
    
def write_to_csv(filename, field_names, data):
    # Check if the file exists
    file_exists = False
    try:
        with open(filename, 'r') as file:   
            file_exists = True
    except FileNotFoundError:
        file_exists = False

    # Open the CSV file in the appropriate mode
    mode = 'a' if file_exists else 'w'
    with open(filename, mode, newline='') as file:
        writer = csv.writer(file)

        # Write a new line if the file is empty
        if not file_exists:
            writer.writerow(field_names)  # Example column headers

        # Write the data to the file
        writer.writerow(data)

def handle_response(response, client_exec_time, server_exec_time, total_inference_time):
    data = pickle.loads(response.body)
    model_name = data['model_name']
    req_id = data['req_id']
    logging.info("Received result of processing request no. "+ str(req_id) + " in batching time = "+ str(data['time_batch'])+ "seconds from server with model: " + str(model_name)+ " and total processing time =  " + str(total_inference_time))
    write_to_csv(client_c+'.csv', metrics_headers, [data['client_id'], data['req_id'], model_name,data['batch'], str(split_index), data['time_batch'], str(client_exec_time), str(server_exec_time), str(total_inference_time)])



async def main():
    model = get_model(model_name, split_index)
    http_client = httpclient.AsyncHTTPClient()
    for req_id in range(int(len(imgs_path)/2)):
        try:
            client_start_time = datetime.datetime.now()
            img = cv2.imread("Images/" + imgs_path[req_id])
            if split_index != 0:
                img = cv2.resize(img,(32,32))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = np.expand_dims(img, axis=0)  # Ensure that input_data has batch dimension
                img = model.predict(img)
            post_data = {'client_id': client_c, 'request_id': req_id + 1, 'image': img, 'model_name': model_name, 'split_index':split_index}
            sending_start_time = datetime.datetime.now()
            serialized_outputs = pickle.dumps(post_data)
            body = base64.b64encode(serialized_outputs)
            response = await http_client.fetch("http://192.168.84.116:8080", method  ='POST', headers = None, body = body)
            receiving_end_time = datetime.datetime.now()

            client_exec_time = (sending_start_time - client_start_time).total_seconds()
            server_exec_time = (receiving_end_time - sending_start_time).total_seconds()
            total_inference_time = (receiving_end_time - sending_start_time).total_seconds()
            #response.add_done_callback(lambda f: handle_response(f.result))
            handle_response(response, client_exec_time, server_exec_time, total_inference_time)
            await tornado.gen.sleep(1)
        except httpclient.HTTPError as e:
            # HTTPError is raised for non-200 responses; the response
            # can be found in e.response.
            print("client send request contains", post_data)
        except Exception as e:
            # Other errors are possible, such as IOError.
            print("Waiting of results...............")
    http_client.close()
    io_loop = ioloop.IOLoop.current()
    io_loop.stop()

if __name__ == '__main__':
    io_loop = ioloop.IOLoop.current()
    io_loop.add_callback(main)
    io_loop.start()