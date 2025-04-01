from __future__ import print_function, absolute_import

from reid.datasets.data.base_dataset import BaseImageDataset

class Image_Layer(BaseImageDataset):
    def __init__(self, image_list, image_list_name, image_list_additional=None, is_train=False, is_query=False, is_gallery=False, verbose=True):
        super(Image_Layer, self).__init__()
        imgs, instructions, pids, cids, cams = [], [], [], [], []  # 将 clothes 重命名为 instructions
        if isinstance(image_list, str):
            with open(image_list, 'r') as f:
                lines = f.readlines()
        else:
            lines = image_list  # image_list 可能是列表（由 merge_sub_datasets 生成）

        for idx, line in enumerate(lines):
            info = line.strip('\n').split(" ")
            if len(info) < 3:
                raise ValueError(f"Line {idx+1} in {image_list_name} has fewer than 3 fields: {line}")
            imgs.append(info[0])
            instructions.append(info[1])  # 占位符，实际文本描述由 PreProcessor 加载
            try:
                pids.append(int(info[2]))
            except (IndexError, ValueError) as e:
                raise ValueError(f"Error parsing person ID in line {idx+1} of {image_list_name}: {line}, error: {e}")
            cids.append(int(info[3]) if len(info) > 3 else 0)
            if len(info) > 4:
                cams.append(int(info[4]))
            elif is_train:
                cams.append(0)
            elif is_query:
                cams.append(-1)
            else:
                cams.append(-2)

        if image_list_additional is not None:
            if isinstance(image_list_additional, str):
                with open(image_list_additional, 'r') as f:
                    lines = f.readlines()
            else:
                lines = image_list_additional
            for idx, line in enumerate(lines):
                info = line.strip('\n').split(" ")
                if len(info) < 3:
                    raise ValueError(f"Line {idx+1} in {image_list_name}_additional has fewer than 3 fields: {line}")
                imgs.append(info[0])
                instructions.append(info[1])
                try:
                    pids.append(int(info[2]))
                except (IndexError, ValueError) as e:
                    raise ValueError(f"Error parsing person ID in line {idx+1} of {image_list_name}_additional: {line}, error: {e}")
                cids.append(int(info[3]) if len(info) > 3 else 0)
                if len(info) > 4:
                    cams.append(int(info[4]))
                elif is_train:
                    cams.append(0)
                elif is_query:
                    cams.append(-1)
                else:
                    cams.append(-2)

        if is_train:
            pids = self._relabel(pids)

        self.data = list(zip(imgs, instructions, pids, cids, cams))
        self.num_classes, self.num_imgs, self.num_cids, self.num_cams = self.get_imagedata_info(self.data)

        if verbose:
            print("=> {} Dataset information has been loaded.".format(image_list_name))
            if is_train:
                self.print_dataset_statistics(self.data, 'train')
            if is_gallery:
                self.print_dataset_statistics(self.data, 'gallery')
            if is_query:
                self.print_dataset_statistics(self.data, 'query')