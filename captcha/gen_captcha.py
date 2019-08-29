from PIL import Image, ImageDraw, ImageFont, ImageFilter
import random
import os
import platform
import numpy as np
import tqdm

class ImgProcess(object):
    # Threshold = 200
    _table = [0 if i < 200 else 1 for i in range(256)]

    @classmethod
    def convert_one_img_file_to_array(cls, img):
        img = img.convert('L')
        img = img.point(cls._table, '1')
        return np.array(img)

#TODO: bold font maybe better
class VerificationCode(object):
    '''用于生成随机验证码'''

    def __init__(self, output_dir):
        self.str_code = list(range(65, 91))
        self.str_code += list(range(97, 123))
        self.str_code += list(range(48, 58))
        self._output_dir = output_dir
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)

        def _prepare_char_2_int_dict():
            # 0->9: 0->9
            # A->Z: 10->26+10
            nums_str = ''.join([chr(i) for i in range(ord('0'), ord('9') + 1)])
            alpha_str = ''.join([chr(i) for i in range(ord('A'), ord('Z') + 1)])
            whole_char = nums_str + alpha_str
            return {c:i for i, c in enumerate(whole_char)}

        self._char_2_int = _prepare_char_2_int_dict()

    def convert_string_to_number_char(self, char_str):
        return [self._char_2_int[c] for c in char_str]
    #  生成随机字符 a~z, A~z, 0~9
    def random_str(self):
        return chr(random.choice(self.str_code))

    # 生成随机颜色:
    def random_color(self):
        return random.randint(0, 245), random.randint(0, 245), random.randint(0, 245)

    def generate_multiple_code_to_zif_file(self, num, file_name):
        height = 22
        weight = 63
        char_num = 5
        imgs = np.zeros((num, height, weight))
        labels = np.zeros((num, char_num))
        for i in tqdm.tqdm(range(num)):
            img, code_str = self.generate_code(char_num)
            img_np = ImgProcess.convert_one_img_file_to_array(img)
            imgs[i] = img_np
            labels[i] = self.convert_string_to_number_char(code_str)

        img_path = os.path.join(self._output_dir, 'img_label.npz')
        img_path_compressed = os.path.join(self._output_dir, file_name)
        np.savez_compressed(img_path_compressed, imgs=imgs, labels=labels)
        print("Generate {} codes to {} OK".format(num, img_path))

    def generate_multiple_code_to_dir(self, num):
        char_num = 5
        for i in tqdm.tqdm(range(num)):
            self.generate_code_and_save(char_num)
    # 生成验证码和图片
    def generate_code(self, char_num=5):
        """add range int between char, as char space here may not equal to target images'"""
        # 22 * 63
        width_per_char = 10
        height = 22

        #width = width_per_char * char_num + 3
        width = 63
        image = Image.new('RGB', (width, height), (255, 255, 255))
        # 根据操作系统获取字体文件
        if platform.uname().system == 'Windows':
            ttf = 'arial.ttf'
        elif platform.uname().system == 'Linux':
            ttf = '/usr/share/fonts/arial/ARIAL.TTF'
        font = ImageFont.truetype(ttf, 12)
        draw = ImageDraw.Draw(image)
        # 输出文字
        code_str = ''
        pos = random.randrange(1, 4)
        for t in range(char_num):
            tmp = self.random_str().upper()
            draw.text((pos, random.randrange(1,4)), tmp, font=font, fill=self.random_color())
            code_str += tmp
            pos += width_per_char + random.randrange(1,4)
        # 模糊处理
        image = image.filter(ImageFilter.EDGE_ENHANCE_MORE)
        return image, code_str

    # 生成验证码和图片
    def generate_code_and_save(self, char_num=5):
        image, code_str = self.generate_code(char_num)
        # 图片保存
        file_name = os.path.join(self._output_dir, '{}.{}'.format(code_str, 'png'))
        image.save(file_name, 'png')
        return code_str

def load_data(imgs_file):
    npz = np.load(imgs_file)
    return npz['imgs'], npz['labels']

if __name__ == '__main__':
    ver_code = VerificationCode('./output')
    ver_code.generate_multiple_code_to_zif_file(60000, 'random_space_imgs')
    #ver_code.generate_multiple_code_to_dir(30)
