import json
from pathlib import Path
import unittest

import cv2
import numpy as np

from core import score_point, validate_config, analyze, read_image
from score_map import attach_map, get_map, near_boundary

ROOT = Path(__file__).resolve().parents[1]


class ScoreMapTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads((ROOT/'target_config.json').read_text(encoding='utf8'))
        self.config.pop('score_map',None)
        attach_map(self.config, ROOT/'assets/Biaso4.png')
        self.labels = cv2.imread(str(ROOT/'Image_Scored/Score.png'), cv2.IMREAD_UNCHANGED)

    def test_each_region_and_background(self):
        for value in [0, 5, 6, 7, 8, 9, 10]:
            points = np.argwhere(self.labels == value)
            for y, x in points[::max(1,len(points)//100)]:
                self.assertEqual(score_point((x,y), self.config), value)

    def test_uses_pixels_even_when_circle_geometry_disagrees(self):
        self.config['center'] = [100,100]
        y,x = np.argwhere(self.labels == 10)[100]
        self.assertEqual(score_point((x,y), self.config), 10)

    def test_outside_and_invalid_coordinates(self):
        for point in [(-.1,450),(450,-.1),(900,450),(450,900)]:
            self.assertEqual(score_point(point,self.config),0)
        with self.assertRaises(ValueError):score_point((float('nan'),0),self.config)
    def test_demo_holes_on_black_pixels_are_zero(self):
        for point in [(174.391647,135.591862),(748.212982,214.166443),(534.058594,20.567027),(251.483269,96.954494)]:
            self.assertEqual(score_point(point,self.config),0)

    def test_snapshot_roundtrip_and_hash_validation(self):
        restored=json.loads(json.dumps(self.config))
        np.testing.assert_array_equal(get_map(restored),self.labels)
        restored['score_map']['sha256']='invalid'
        with self.assertRaises(ValueError):validate_config(restored)

    def test_dimensions_and_confirmation(self):
        self.assertTrue(self.config['confirmed'])
        self.config['confirmed']=True
        attach_map(self.config,ROOT/'assets/Biaso4.png')
        self.assertTrue(self.config['confirmed'])
        self.config['image_size']=[1,1]
        with self.assertRaises(ValueError):validate_config(self.config)

    def test_boundary_uses_pixel_regions(self):
        y,x=np.argwhere(self.labels[:,1:] != self.labels[:,:-1])[100]
        self.assertTrue(near_boundary((x,y),self.labels,2))
        self.assertFalse(near_boundary((451,453),self.labels,2))

    def test_pipeline_scores_in_aligned_coordinates(self):
        image=read_image(ROOT/'assets/Biaso4.png')
        points=[]
        for value in range(5,11):
            y,x=np.argwhere(self.labels==value)[100]
            points.append([int(x),int(y)])
        class Detector:
            calls=0
            def detect(self,image,confidence):
                self.calls+=1
                return [] if self.calls==1 else [dict(point=p,confidence=.99) for p in points]
        corners=[[0,0],[899,0],[899,899],[0,899]]
        result=analyze(image,image,image,self.config,Detector(),pairs=[(corners,corners)]*2)
        self.assertEqual([h['score'] for h in result['detections']],list(range(5,11)))


if __name__=='__main__':unittest.main()
