import unittest
import numpy as np
from PIL import Image
from split_dataset import scene_group, split_rows
from train_cloudvision import TinyUNet, letterbox_rgb_and_mask
import torch

class TrainingStageTests(unittest.TestCase):
    def test_grouping(self):
        a = scene_group("2025-09-18_12-14-01_SXC3-234_1.jpg")
        b = scene_group("2025-09-18_12-14-44_SXC3-234_1.jpg")
        self.assertEqual(a, b)

    def test_group_split_no_leakage(self):
        rows = []
        for i in range(12):
            for j in range(2):
                rows.append({"image": f"2025-09-{i+1:02d}_12-11-{j:02d}_SXC3-10{i}_1.jpg",
                             "manual_cloud_percent": "92.0" if i % 3 == 0 else "23.0"})
        assignment = split_rows(rows)
        group_to_split = {}
        for r in assignment:
            self.assertTrue(r['split'] in ('train','val','test'))
            if r['group'] in group_to_split:
                self.assertEqual(group_to_split[r['group']],r['split'])
            group_to_split[r['group']] = r['split']
        self.assertEqual(set(group_to_split.values()),{'train','val','test'})

    def test_letterbox_preserves_aspect_and_mask(self):
        pic = Image.new('RGB', (400, 200), 'white')
        label = Image.new('L',(100, 50),255)
        a, b, valid = letterbox_rgb_and_mask(pic,label,192)
        self.assertEqual(a.size,(192,192))
        self.assertEqual(np.asarray(valid).sum()//255, 192*96)
        self.assertEqual(np.asarray(b).sum()//255, 192*96)

    def test_unet_output_shape(self):
        net = TinyUNet()
        with torch.no_grad():
            out = net(torch.zeros(1,3,64,64))
        self.assertEqual(tuple(out.shape),(1,1,64,64))

if __name__ == '__main__':
    unittest.main()
