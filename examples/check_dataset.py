from reid.datasets.data_builder_t2i import merge_sub_datasets

def check_dataset(query_list, gallery_list, root):
    class Args:
        def __init__(self):
            self.query_list = query_list
            self.gallery_list = gallery_list
            self.root = root
            self.data_config = {'json_file': r'D:\Instruct-ReID\cuhk_pedes\annotations\annotations.json'}

    args = Args()
    # 加载 query 数据
    query_lines = merge_sub_datasets(args.query_list, args.root, args)
    query_data = [(img_path, caption, int(pid), cam_id) for img_path, caption, pid, _, cam_id in query_lines]
    query_ids = set([pid for _, _, pid, _ in query_data])
    print(f"Query set: {len(query_data)} samples, {len(query_ids)} unique IDs")

    # 加载 gallery 数据
    gallery_lines = merge_sub_datasets(args.gallery_list, args.root, args)
    gallery_data = [(img_path, caption, int(pid), cam_id) for img_path, caption, pid, _, cam_id in gallery_lines]
    gallery_ids = set([pid for _, _, pid, _ in gallery_data])
    print(f"Gallery set: {len(gallery_data)} samples, {len(gallery_ids)} unique IDs")

    # 检查身份交集
    total_ids = query_ids.union(gallery_ids)
    print(f"Total unique IDs in test set: {len(total_ids)}")
    print(f"Intersection between query and gallery IDs: {len(query_ids.intersection(gallery_ids))}")

if __name__ == "__main__":
    query_list = r"D:\Instruct-ReID\cuhk_pedes\splits\query_t2i_v2.txt"
    gallery_list = r"D:\Instruct-ReID\cuhk_pedes\splits\gallery_t2i_v2.txt"
    root = r"D:\Instruct-ReID\cuhk_pedes\images"
    check_dataset(query_list, gallery_list, root)