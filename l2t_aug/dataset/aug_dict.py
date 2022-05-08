from dataset.data_transforms import *
  
def make_aug_dict():
   
    aug_dict={0:ShearX(0.1),1:ShearX(0.2),2:ShearX(0.3),3:ShearY(0.1),4:ShearY(0.2),5:ShearY(0.3),6:TranslateX(0.15),
    7:TranslateX(0.3),8:TranslateX(0.45),9:TranslateY(0.15),10:TranslateY(0.3),11:TranslateY(0.45),12:Rotate(10),
    13:Rotate(20),14:Rotate(30),15:Color(0.3),16:Color(0.6),17:Color(0.9),18:Posterize(4),19:Posterize(5),20:Posterize(8),21:Solarize(26),22:Solarize(102),23:Solarize(179),
    24:Contrast(1.3),25:Contrast(1.6),26:Contrast(1.9),27:Sharpness(1.3),28:Sharpness(1.6),29:Sharpness(1.9),30:Brightness(1.3),31:Brightness(1.6),32:Brightness(1.9),
    33:AutoContrast(),34:Equalize(),35:Invert()}
    index_start = 0
    final_dict = {}
    for k_1 in aug_dict.keys():
        for k_2 in aug_dict.keys():
            final_dict[index_start] = (aug_dict[k_1],aug_dict[k_2])
            index_start += 1 
    return final_dict 