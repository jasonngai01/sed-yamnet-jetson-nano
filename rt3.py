#import the lib for deep learning


print('Loading TensorFlow...')
import numpy as np
#import scipy.signal
import soundfile as sf
import tensorflow as tf

print('Loading UAVnet...')
import params as yamnet_params
import yamnet as yamnet_model
params = yamnet_params.Params()
yamnet = yamnet_model.yamnet_frames_model(params)
yamnet.load_weights('yamnet.h5')
yamnet_classes = yamnet_model.class_names('yamnet_class_map.csv')

#load UAV model >> tflite file
# using the colab https://colab.research.google.com/drive/12gole2MzX8SM2omJKLHh3YjjRe70he6V to train the model
interpreter = tf.lite.Interpreter(r'model\uav20210913.tflite')
input_details = interpreter.get_input_details()
waveform_input_index = input_details[0]['index']
output_details = interpreter.get_output_details()
scores_output_index = output_details[0]['index']


import os, pyaudio, time
#os.system('jack_control start')
p = pyaudio.PyAudio()
os.system('clear')
#print('Sound Event Detection by running inference on every 1.024 second audio stream from the microphone!\n')

CHUNK = 1024 # frames_per_buffer # samples per chunk
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
RECORD_SECONDS = 1.024                                          # need at least 975 ms
INFERENCE_WINDOW = 2 * int(RATE / CHUNK * RECORD_SECONDS)       # 2 * 16 CHUNKs
THRESHOLD = 0.4

stream = p.open(format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        frames_per_buffer=CHUNK)

CHUNKs = []
with open('sed.npy', 'ab') as f:
    while True:
        try:
            stream.start_stream()
            for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
                data = stream.read(CHUNK)
                CHUNKs.append(data)
                # print(len(CHUNKs))
            stream.stop_stream()

            if len(CHUNKs) > INFERENCE_WINDOW:
                CHUNKs = CHUNKs[int(RATE / CHUNK * RECORD_SECONDS):]
                # print('new len: ',len(CHUNKs))
            wav_data = np.frombuffer(b''.join(CHUNKs), dtype=np.int16)
            waveform = wav_data / tf.int16.max#32768.0
            waveform = waveform.astype('float32')

            scores, embeddings, spectrogram = yamnet(waveform)

            # model prediction using uavnet tflite
            interpreter.resize_tensor_input(waveform_input_index, [waveform.size], strict=True)
            interpreter.allocate_tensors()
            interpreter.set_tensor(waveform_input_index, waveform)
            interpreter.invoke()
            scores = interpreter.get_tensor(scores_output_index)
            top_class_index = scores.argmax()
            
            print(time.ctime().split()[3])

            #result in terminal
            if top_class_index == 1:
                print('UAV detected')

            else:         
                print('NO UAV detected')

            prediction = np.mean(scores[:-1], axis=0) # last one scores comes from insufficient samples
            assert (prediction==scores[0]).numpy().all() # only one scores at RECORD_SECONDS = 1.024
            assert len(scores[:-1]) == CHUNK * len(CHUNKs) / RATE // 0.48 - 1 # hop 0.48 seconds
            top5 = np.argsort(prediction)[::-1][:5]
            print(time.ctime().split()[3],
                ''.join((f" {prediction[i]:.2f} 👉{yamnet_classes[i][:7].ljust(7, '　')}" if prediction[i] >= THRESHOLD else '') for i in top5))
            np.save(f, np.concatenate(([time.time()], prediction)))
        except:
            stream.stop_stream()
            stream.close()
            p.terminate()
            f.close()
