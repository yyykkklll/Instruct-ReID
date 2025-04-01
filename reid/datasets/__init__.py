from .data_builder_t2i import DataBuilder_t2i



def dataset_entry(this_task_info):
    return globals()[this_task_info.task_name]