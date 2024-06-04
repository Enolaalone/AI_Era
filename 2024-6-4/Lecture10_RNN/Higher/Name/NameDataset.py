import csv
import gzip
from torch.utils.data import Dataset  # Dataset DataLoader


class NameDataset(Dataset):
    def __init__(self, is_train_set=True):
        filename = '../names_train.csv.gz' if (is_train_set) else '../names_test.csv.gz'
        with gzip.open(filename, 'rt') as f:
            reader = csv.reader(f)
            rows = list(reader)  # 转为列表
        self.names = [row[0] for row in rows]  # 从文件取出含(name,country)元组中的name
        self.len = len(self.names)  # name长度

        # country
        self.countries = [row[0] for row in rows]  # 从文件取出含(name,country)元组中的country
        self.country_lists = list(sorted(set(self.countries)))  # 转为国家数列 ；排序 ；结合去除重复
        self.country_dict = self.get_country_dict()  # 获取国家序号字典
        self.countries_nums = len(self.country_lists)  # 国家数

    def __getitem__(self, idx):
        return self.names[idx], self.country_dict[self.countries[idx]]  # 输入name 与 输出country

    def __len__(self):
        return self.len

    def get_country_dict(self):  # 国家字典
        country_dict = dict()
        for idx, country in enumerate(self.country_lists):
            country_dict[country] = idx
        return country_dict

    def get_country_name(self, index):  # 用索引返回国家名称
        return self.country_lists[index]

    def get_country_num(self):  # 返回国家数目
        return self.countries_nums
