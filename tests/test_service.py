import io
import os
import unittest
from unittest.mock import patch
import numpy as np
from PIL import Image
from fastapi import HTTPException
from fastapi.testclient import TestClient
from backend.service import app, ModelRunner, preprocess, read_image

class ServiceTests(unittest.TestCase):
    def test_missing_checkpoint_health_and_prediction(self):
        with patch.dict(os.environ, {'GRNET_CHECKPOINT': ''}):
            state=ModelRunner().status()
        self.assertFalse(state['ready'])
        client=TestClient(app)
        self.assertFalse(client.get('/health').json()['ready'])
        buf=io.BytesIO();Image.fromarray(np.zeros((4,4,3),dtype=np.uint8)).save(buf,format='PNG')
        response=client.post('/predict',files={'image':('image.png',buf.getvalue(),'image/png'),'ndvi':('ndvi.png',buf.getvalue(),'image/png')})
        self.assertEqual(response.status_code,503)
        self.assertNotIn('green_mask',response.json())
    def test_authentication(self):
        with patch.dict(os.environ, {'API_TOKEN':'test-secret'}):
            client=TestClient(app)
            self.assertEqual(client.get('/health').status_code,401)
            self.assertEqual(client.get('/health',headers={'Authorization':'Bearer test-secret'}).status_code,200)
    def test_five_channel_training_normalization(self):
        image=np.zeros((2,3,4),dtype=np.uint8);image[:,:,0]=255;image[:,:,1]=128;image[:,:,2]=64;image[:,:,3]=192
        ndvi=np.array([[0,1,-1],[0,0,1]],dtype=np.float32)
        tensor=preprocess(image,ndvi)
        self.assertEqual(tensor.shape,(5,2,3))
        np.testing.assert_allclose(tensor[:,0,0],[1,128/255,64/255,192/255,.5])
        np.testing.assert_allclose(tensor[4,0],[.5,1,0])
    def test_rgba_is_not_nir(self):
        buf=io.BytesIO();Image.fromarray(np.ones((2,3,4),dtype=np.uint8)*255).save(buf,format='PNG')
        image=read_image(buf.getvalue(),'image.png')
        self.assertEqual(image.shape,(2,3,3))
        with self.assertRaises(HTTPException):preprocess(image,np.zeros((2,3),dtype=np.uint8))
    def test_invalid_dn_and_ndvi(self):
        with self.assertRaises(HTTPException):preprocess(np.zeros((2,3,4),dtype=np.uint16),np.zeros((2,3),dtype=np.uint8))
        with self.assertRaises(HTTPException):preprocess(np.zeros((2,3,4),dtype=np.uint8),np.ones((2,3),dtype=np.uint8)*255)
        with self.assertRaises(HTTPException):preprocess(np.zeros((2,3,4),dtype=np.uint8),np.zeros((2,4),dtype=np.uint8))

if __name__=='__main__':unittest.main()
