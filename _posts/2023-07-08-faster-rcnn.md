---
layout: post
title:  Faster R-CNN - Phương pháp phát hiện đối tượng 02 giai đoạn và lịch sử
date:   2023-07-08 13:50:00
description: Faster R-CNN - Phương pháp phát hiện đối tượng 02 giai đoạn và lịch sử
tags: formatting links
categories: sample-posts
inline: false
---

## Lời dẫn 

Phát hiện đối tượng là một trong các bài toán cơ sở của lĩnh vực Thị giác máy tính, và hiện vẫn được nghiên cứu rất sôi nổi. Mỗi năm, tại các hội nghị lớn như CVPR, ICCV, ECCV, ICLR đều xuất hiện các công bố liên quan đến bài toán này. Nhận thấy phát hiện đối tượng là bài toán quan trọng và là nền tảng giúp phát triển các bài toán khác trong thực tế, nhóm sinh viên lựa chọn và thực hiện tìm hiểu về bài toán này.

## Các nghiên cứu trước khi có Faster R-CNN

Hiện nay, các phương pháp phát hiện đối tượng được phân thành 02 loại: 02 giai đoạn và 01 giai đoạn. Đối với 01 giai đoạn sẽ được chia thành 02 nhánh phương pháp khác: anchor-free và anchor-based. Tuy nhiên, trong nội dung tiểu luận này, nhóm sinh viên chỉ tìm hiểu về phương pháp Faster R-CNN, và phương pháp này thuộc nhóm phương pháp 02 giai đoạn. Trước khi Faster R-CNN ra đời, đã có sự xuất hiện của 02 phương pháp: R-CNN và Fast R-CNN, trong đó Fast R-CNN là một cải tiến vô cùng hiệu quả của R-CNN, và Faster R-CNN là một bước phát triển lớn từ Fast R-CNN. Để trình bày Faster R-CNN, nhóm sinh viên sẽ nói lại sơ qua về phương pháp R-CNN và Fast R-CNN.

### R-CNN

Vào thời điểm trước khi nhóm phương pháp R-CNN ra đời, các phương pháp phát hiện đối tượng thường dựa vào đặc điểm của ảnh như màu sắc, và các thuật toán sử dụng mang hơi hướng phân cụm. R-CNN ra đời đánh dấu kỷ nguyên sử dụng kỹ thuật học sâu cho bài toán phát hiện đối tượng.

![](https://i.imgur.com/wxFZHHH.png)
Hình 1. Minh họa phương pháp R-CNN [1].

Phương pháp này có thể được mô tả đơn giản như sau: đầu tiên thuật toán **selective search** sẽ chọn ra khoảng $N$ vùng trên ảnh có khả năng cao chứa đối tượng. Thuật toán này chủ yếu dựa vào các đặc điểm bức ảnh như màu sắc. Trong bài báo gốc, các tác giả sử dụng $N = 2000$, tức sẽ có $2000$ vùng được đề xuất trên ảnh. Từ 2000 vùng này, ta sẽ tiến hành cắt ra từ ảnh gốc, và một mạng CNN sẽ được sử dụng để trích xuất đặc trưng của 2000 vùng ảnh này. Sau đó, từ lớp đặc trưng cuối cùng sẽ đi qua 1 lớp FC để tính toán một bộ offset $(\delta x, \delta y, \delta w, \delta h)$, trong đó $(\delta x, \delta y)$ là offset tọa độ tâm của đối tượng, $(\delta w, \delta h)$ là offset chiều rộng và chiều cao của đối tượng. Như vậy, mạng sẽ học cách bo sát đối tượng và phân lớp đối tượng từ N vùng truyền vào ban đầu. Đối tượng sẽ được phân lớp bằng thuật toán SVM (Support Vector Machine). 

> Q: offset là gì?

> A: offset gọi là phần bù. Tức là ban đầu selective search chọn ra 2000 vùng. 2000 vùng này đều có tọa độ (x, y, w, h). Tuy nhiên nó chưa bo sát đối tượng, ta cần một nhánh FC học cách bo sát, nhưng ta không học ra tọa độ chính xác, mà từ tọa độ ở selective search ta căn chỉnh lại, đó gọi là offset.

Vậy ở đây chúng ta thấy điều gì? R-CNN phải thực hiện trích xuất đặc trưng cho $2000$ vùng ảnh, như thế rất tốn thời gian. Cuộc sống luôn phải vận động và phát triển, do đó Fast R-CNN ra đời để khắc phục điểm yếu chí mạng này của R-CNN.

### Fast R-CNN

Về cơ bản, Fast R-CNN vẫn giữ ý tưởng 02 giai đoạn của R-CNN. Selective Search vẫn được sử dụng để tìm 2000 vùng có khả năng chứa đối tượng, tuy nhiên điểm khác biệt ở đây là Fast R-CNN **trích xuất đặc trưng ảnh trước**, sau đó mới sử dụng selective search để tìm ra 2000 vùng đặc trưng. Do đó, ta chỉ cần đưa 2000 vùng đặc trưng này để tiếp tục xử lý mà không cần rút trích lại. Điều này là một bước ngoặt, vì nó đã tăng tốc độ xử lý lên rất nhiều ($\times 18.3$ cho huấn luyện và $\times 146$ cho suy luận) nhưng vẫn giữ được độ chính xác.

![](https://i.imgur.com/ZBggmRI.png)
Hình 2. Minh họa mô hình Fast R-CNN [2].

Một điều cải tiến khác, Fast R-CNN không dùng SVM để phân lớp đối tượng nữa, mà tác giả gắn vào mạng một đầu FC để phân lớp. Tức là bây giờ chúng ta có 02 lớp FC, 1 lớp FC để phân lớp đối tượng, 1 lớp FC còn lại để tính toán bộ offset để căn chỉnh tọa độ đối tượng lại cho chính xác (giống như bên R-CNN).

Cùng xem lại bảng kết quả so sánh giữa R-CNN và Fast R-CNN. Có thể thấy 

![](https://i.imgur.com/IrW35vU.png)

Hình 3. Kết quả so sánh giữa Fast R-CNN và R-CNN [2].

Hình 3 là hình chụp kết quả từ bài báo gốc của Fast R-CNN. Có thể thấy rằng Fast R-CNN có kết quả tương đương với R-CNN, nhưng tốc độ vượt trội hơn rất nhiều. Tuy nhiên ý tưởng sử dụng 1 lớp FC để phân lớp chỉ vượt trội ở việc phát hiện các đối tượng lớn (L), các loại đối tượng nhỏ (S), trung bình (M) có hiệu quả phát hiện tương đương như R-CNN. Tuy nhiên với tốc độ vượt trội thế này thì không có lý do gì phải dùng R-CNN nữa.

Tuy nhiên, Fast R-CNN vẫn có điểm yếu chí mạng của nó. Đó là việc lựa chọn vùng đề xuất vẫn dựa vào thuật toán Selective Search. *Liệu có cách nào cải tiến chính xác hơn ở điểm này hay không? Một cách tiếp cận khác mà nó tự học để đề xuất luôn.*

Trên đây nhóm sinh viên cung cấp một cái nhìn tương đối đầy đủ về hoàn cảnh ra đời của Faster R-CNN. Phần dưới đây nhóm sinh viên sẽ trình bày kỹ hơn về Faster R-CNN.

## Faster R-CNN

Khắc phục điểm yếu chí mạng của Fast R-CNN, Faster R-CNN ra đời với hai điểm nổi bật duy nhất:
- Mạng đề xuất khu vực (Regional Proposal Network), gọi tắt là RPN. Mạng này được huấn luyện để phát hiện các vùng trên ảnh khả năng cao chứa đối tượng. Nói một cách đơn giản, RPN sẽ thay thế Selective Search.
- Cơ chế chia sẻ trọng số giữa RPN và Fast R-CNN.

Đơn giản dễ hiểu, Faster R-CNN là sự kết hợp của RPN và mạng Fast R-CNN đã trình bày ở phía trên. Sau đó các tác giả đã thử các cơ chế chia sẻ trọng số khác nhau để kết hợp hoàn hảo hai thứ này. Dưới đây chúng ta sẽ tìm hiểu kỹ về mạng đề xuất khu vực. Đáng chú ý, Faster R-CNN được công bố tại hội nghị NeurlPS, là một hội nghị có những công bố mang tính bước ngoặt. Vì lẽ đó, Faster R-CNN được coi là một bước ngoặt. Faster R-CNN không còn là một phương pháp, Faster R-CNN đã là một hệ tư tưởng khi nhắc về nhóm phương pháp 02 giai đoạn. Dưới đây nhóm sinh viên sẽ trình bày kỹ về mạng đề xuất khu vực, và cơ chế chia sẻ trọng số - những thứ làm nên thành công của Faster R-CNN.

### Mạng đề xuất khu vực (Regional Proposal Network)

Như đã nói, mạng đề xuất khu vực sẽ được dùng để đề xuất các vùng khả năng cao chứa đối tượng thay cho thuật toán Selective Search. Gọi nó là "mạng" vì nó có "học". Thế làm sao để chúng ta huấn luyện được mạng này?

![](https://i.imgur.com/POeFrpv.png)

Hình 4. Minh họa cách chọn mẫu của Faster R-CNN [3].

Để huấn luyện được mạng RPN, chúng ta cần tạo dữ liệu cho nó học, cơ chế này được gọi là chọn mẫu (sampling). Để giải thích về cơ chế này, dễ nhất các bạn đọc quan sát Hình 4. Từ đặc trưng của ảnh (conv feature map), sẽ có một cửa sổ trượt có kích thước $N \times N$ (sliding window) trượt qua một cách lần lượt. Tại mỗi lần trượt, nó sẽ tạo ra $k$ anchor box.

> Q: khoan khoan, dừng lại khoảng chừng là 2 giây. Anchor box là gì vậy?
> A: anchor box là một khái niệm chỉ các "hộp neo" được định nghĩa trước, làm tiền đề cho các bước phía sau. Tại sao gọi là "hộp neo", vì nó sẽ tạm thời neo đậu, dựa vào các hộp neo đậu này mà bằng một cách nào đó chúng ta sẽ sử dụng nó để xác định đối tượng trên ảnh. Khái niệm này sẽ còn gặp lại ở các nhóm phương pháp 01 giai đoạn.

$k$ anchor box có $k$ kích thước khác nhau, trong bài báo gốc, tác giả chọn $k=9$. Mỗi anchor box sẽ được gán 2 thứ:

- Lớp nhị phân xác định nó có đối tượng hay không. Nếu anchor box đó được xác định là mẫu có chứa đối tượng, ta gán là $1$, ngược lại là $0$.
- Tọa độ của anchor box, bao gồm 04 phần tử \(x, y, w, h\).

Thế thì làm sao chúng ta xác định được liệu 1 anchor box có chứa đối tượng hay không? Câu trả lời đơn giản là ta đi so với mẫu dữ liệu thật. Mỗi 1 anchor box sẽ được đi tính toán độ đo IoU với các hộp dự đoán thật (ground-truth), sau đó ta sẽ lấy giá trị IoU cao nhất của anchor box đó so với các ground-truth. Anchor box đó sẽ được chọn là mẫu $1$ khi $IoU > 0.7$, và chọn là mẫu $0$ khi $IoU \leq 0.3$. Vậy còn một đoạn từ \(0.3, 0.7\] ta sẽ không chọn.

Như vậy, bây giờ ta đã có một tập dữ liệu bao gồm các anchor box có đối tượng và anchor box không đối tượng cho mạng RPN học.

Để huấn luyện ta cần định nghĩa một hàm mất mát tối ưu hóa đa mục tiêu: 1) xác định có hoặc không chứa đối tượng (phân lớp nhị phân); 2) hồi quy tọa độ. Hàm mất mát để huấn luyện RPN được định nghĩa như sau:

$$L(\{p_i\}, \{t_i\}) = \frac{1}{N_{cls}} \sum_i L_{cls} (p_i, \hat p_i) + \lambda \frac{1}{N_{reg}} \sum_i \hat p_i L_{reg}(t_i, \hat t_i)$$

Trong đó, $L_{cls}$ là hàm binary cross-entropy, còn $L_{reg}$ là hàm mất mát hồi quy $SmoothL1$; $\hat p_i$ là phân phối xác suất dự đoán hộp đề xuất có đối tượng hay không, $p_i$ là nhãn thật sự; $\hat t_i$ là tọa độ dự đoán đã được tham số hóa; $t_i$ là tọa độ thật đã được tham số hóa, $t=\{t_x, t_y, t_w, t_h\}$ và $\hat t=\{\hat t_x, \hat t_y, \hat t_w, \hat t_h\}$ được định nghĩa như sau:

$$ t_x = (x - x_a) / w_a; \hat t_x = (\hat x - x_a) / w_a $$
$$ t_y = (y - y_a) / h_a; \hat t_y = (\hat y - y_a) / h_a $$
$$ t_w = \log (w / w_a); \hat t_w = \log (\hat w / w_a) $$
$$ t_h = \log (h / h_a); \hat t_h = \log (\hat h / h_a) $$

Trong đó, $x$, $\hat x$, $x_a$ lần lượt là hoành độ tâm thật sự của hộp bao, hoành độ tâm dự đoán của hộp bao và hoành độ tâm anchor box. $y$, $\hat y$, $y_a$ là tung độ tâm. Như thế ta có thể thấy, ở đây RPN không cố gắng để dự đoán các vị trí giống như hộp mỏ neo (vì nếu thế thì sử dụng hộp mỏ neo luôn cần gì học nữa), mà nó sẽ cố gắng dự đoán các vùng đề xuất lân cận các anchor box, mà tại đó có nhiều hộp bao ground-truth.

Thực tế, RPN sinh ra rất nhiều hộp đề xuất, trong khi chúng ta chỉ cần $N=2000$ hộp đề xuất. Có nhiều hộp đề xuất ban đầu bị chồng lấp lên nhau, lúc này tác giả sử dụng thuật toán non maximum suppression (NMS) với ngưỡng $IoU=0.7$ để loại đi các hộp đề xuất chồng lấp, chỉ giữ lại $2000$ hộp đề xuất để đưa vào Fast R-CNN huấn luyện. Tuy nhiên, khi huấn luyện thì $N=2000$, trong khi đó lúc đánh giá (test) thì $N$ là một con số khác (nhỏ hơn 2000).

### Cơ chế chia sẻ trọng số giữa RPN và Fast R-CNN (Sharing Convolutional Features)

Việc huấn luyện RPN và Fast R-CNN được thực hiện độc lập chứ không end-to-end. Lý do của việc này rất dễ hiểu: rất khó để huấn luyện hai thứ này cùng một lúc. Fast R-CNN cần có một cơ chế sinh mẫu cố định (sinh ra các hộp đề xuất) để học, và bản thân Fast R-CNN cũng có một hàm tối ưu hóa tương tự như RPN (tổng của hàm mất mát phân lớp đối tượng và hàm mất mát hồi quy, nhưng hàm mất mát ở Fast R-CNN dùng cho đa lớp). Do đó không biế rằng nếu cơ chế sinh mẫu liên tục thay đổi (liên tục cập nhật trọng số lại ở RPN) thì mạng hợp nhất này có hội tụ được hay không. Do đó, tác giả đã đề xuất một cơ chế huấn luyện chia sẻ trọng số gồm 4 bước sau đây:

- Bước 1: Huấn luyện RPN trước
- Bước 2: Đóng băng trọng số ở RPN, sử dụng mạng RPN vừa học được sinh ra các vùng đề xuất cho mạng Fast R-CNN.
- Bước 3: Dùng trọng số vừa học được ở mạng Fast R-CNN làm trọng số khởi tạo của mạng RPN ở các lớp tích chập (RPN và Fast R-CNN chia sẻ trọng số ở một số lớp tích chập), và chỉ fine-tune các lớp tích chập của riêng FPN và 2 lớp FC.
- Bước 4: Đóng băng trọng số ở mạng RPN vừa học để sinh các mẫu cho Fast R-CNN, và chỉ fine-tune Fast R-CNN ở 2 lớp FC.

### Kết quả so với Fast R-CNN

![](https://i.imgur.com/cV1APRP.png)

Hình 5. Kết quả so sánh giữa việc sử dụng RPN và thuật toán Selective Search [3].

Có thể thấy, việc sử dụng thuật toán Selective Search cho kết quả thấp hơn khi sử dụng RPN. Hơn nữa, tốc độ xử lý của Fast R-CNN sử dụng RPN (Faster R-CNN) cũng nhanh hơn sử dụng thuật toán Selective Search. Vừa tốt hơn vừa nhanh hơn, Faster R-CNN đã trở thành một "hệ tư tưởng".

## Hướng dẫn huấn luyện thử Faster R-CNN

Faster R-CNN khá khó để có thể code from scratch. Do đó, trong nội dung bài viết này, nhóm sinh viên hướng dẫn bạn đọc cách huấn luyện thử một mô hình Faster R-CNN thông qua toolbox MMDetection [4].

Lưu ý: source code được thực nghiệm trên Google Colab.

Trong bài này, nhóm sinh viên sử dụng bộ dữ liệu UIT-VinaDeveS22 được cung cấp ở đây: https://github.com/nguyenvd-uit/uit-together-dataset/blob/main/UIT-VinaDeveS22.md

UIT-VinaDeveS22 là bộ dữ liệu phát hiện phương tiện giao thông từ camera CCTV, các phương tiện trong bộ dữ liệu bao gồm: bicycle, motorcycle, car, van, truck, bus, fire truck.

![](https://i.imgur.com/xZDJryF.png)
Hình 6. Hình ảnh bộ dữ liệu UIT-VinaDeveS22

### 1. Cài đặt thư viện

Đầu tiên, chúng ta cần clone toolbox MMDetection từ Github về, sau đó cài đặt các thư viện cần thiết:


- Bước 1.1: git clone thư mục code
```[python3]
!git clone https://github.com/open-mmlab/mmdetection
```

- Bước 1.2: cài đặt thư viện
```
# Cài đặt thư viện
%cd mmdetection
!pip install -r requirements.txt
!pip install -v -e .
!pip install mmcv-full==1.3.8 -f https://download.openmmlab.com/mmcv/dist/cu101/torch1.7.0/index.html
!pip install torch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2
```

### 2. Import thư viện

- Bước 2.1: ta import các thư viện liên quan cần dùng, ở đây chúng ta cần import `mmdet`, `mmcv`.

```[python3]
# Dùng để build config
import mmdet
from mmdet.apis import set_random_seed
from mmcv import Config

# Dùng để xây dựng dataset
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.apis import train_detector

# Dùng để dự đoán
from mmdet.apis import init_detector, inference_detector, show_result_pyplot
import mmcv
```

### 3. Chuẩn bị dữ liệu

Tải dữ liệu tại đây: https://drive.google.com/file/d/1NhsIWyPdqF2KDqPWU926eZwCLk0gtjnv/view?usp=sharing

Ta có thể tải bằng lệnh sau:

```
!gdown --id 1NhsIWyPdqF2KDqPWU926eZwCLk0gtjnv
```

Sau khi tải, chúng ta giải nén dữ liệu bằng lệnh `unzip`

```
!unzip UIT-VinaDeveS22.zip
```

Bộ dữ liệu được tổ chức theo cấu trúc sau:

```
UIT-VinaDeveS22
|__ images
|____ *.jpg
|__ outputtrain.json
|__ outputvalid.json
|__ outputtest.json
```

Thư mục `images` chứa toàn bộ ảnh của bộ dữ liệu. 3 file `ouputtrain.json`, `outputvalid.json` và `outputtest.json` là 3 file annotation đã được chuẩn bị theo định dạng MS-COCO ứng với 3 tập: train, valid, test.

### 4. Chuẩn bị config

Bước này khá quan trọng. Hiện tại MMDetection đã hỗ trợ cho chúng ta rất nhiều config của nhiều phương pháp SOTA cho phát hiện đối tượng hiện tại, lên tới vài chục phương pháp. Tuy nhiên, chúng ta chỉ sử dụng config dành cho Faster R-CNN.

- Bước 4.1: Config trong MMDetection được thiết kế theo cơ chế kế thừa, tức là từ config chuẩn bị sẵn, chúng ta sẽ tinh chỉnh cho phù hợp với bộ dữ liệu của chúng ta. Đầu tiên ta cần xác định config cha mà chúng ta sẽ kế thừa

```[python3]
cfg = Config.fromfile('configs/faster_rcnn/faster_rcnn_r101_fpn_1x_coco.py')
```

`faster_rcnn_r101_fpn_1x_coco.py` là file config của phương pháp Faster R-CNN, sử dụng kiến trúc CNN ResNet-101 để trích xuất đặc trưng ảnh (là các lớp conv chung của mạng RPN và Fast R-CNN). Trong đó có sử dụng FPN (một cơ chế multi-scale feature map, tức là thay vì qua mạng CNN chỉ có 1 đầu ra thì nó sẽ có nhiều đầu ra với các resolution khác nhau).

- Bước 4.2: Ta cần định nghĩa lại dữ liệu. MMDetection hỗ trợ các bộ dữ liệu sẵn có như COCO, sử dụng bộ dữ liệu khác ta cần định nghĩa lại các lớp. Tuy nhiên do cấu trúc bộ dữ liệu cũng giống như COCO, ta chỉ kế thừa nó về chỉnh lại các lớp sao cho phù hợp:

```[python3]
from mmdet.datasets.builder import DATASETS
from mmdet.datasets import CocoDataset

@DATASETS.register_module()
class VinaDeveS22(CocoDataset):
    CLASSES = ('bicycle', 'motorcycle', 'car', 'van', 'truck', 'bus', 'fire truck')
```

- Bước 4.3: ta đặt đường dẫn cho tập dữ liệu, bao gồm đường dẫn ảnh, annotation cho tập train, test, valid:

```[python3]
# Đường dẫn dữ liệu
import os
data_root_dir = '/content/drive/MyDrive/BDL_UIT/UIT-VinaDeveS22/'

# Chuẩn bị config

cfg = Config.fromfile('configs/faster_rcnn/faster_rcnn_r101_fpn_1x_coco.py')

# Modify dataset type and path
cfg.data_root = data_root_dir

cfg.data.test.type = 'VinaDeveS22'
cfg.data.test.ann_file = os.path.join(data_root_dir, 'outputtest.json')
cfg.data.test.img_prefix = os.path.join(data_root_dir, 'images')

cfg.data.train.type = 'VinaDeveS22'
cfg.data.train.ann_file = os.path.join(data_root_dir, 'outputtrain.json')
cfg.data.train.img_prefix = os.path.join(data_root_dir, 'images')

cfg.data.val.type = 'VinaDeveS22'
cfg.data.val.ann_file = os.path.join(data_root_dir, 'outputvalid.json')
cfg.data.val.img_prefix = os.path.join(data_root_dir, 'images')
```

- Bước 4.4: Một số cấu hình khác như số epoch để save checkpoint 1 lần, learning rate, đường dẫn lưu checkpoint, số lớp cần classify của mô hình, ...

```[python3]
# Một số cấu hình khác

cfg.optimizer.lr = 0.02 / 8
cfg.lr_config.warmup = None
cfg.log_config.interval = 500

# We can set the evaluation interval to reduce the evaluation times
cfg.evaluation.interval = 1
# We can set the checkpoint saving interval to reduce the storage cost
cfg.checkpoint_config.interval = 3

# Set seed thus the results are more reproducible
cfg.seed = 0
set_random_seed(0, deterministic=False)
cfg.gpu_ids = range(1)

# Số class
cfg.model.roi_head.bbox_head.num_classes = 7

# Đường dẫn lưu checkpoints
cfg.work_dir = './checkpoints'
```

### 5. Xây dựng model và huấn luyện

Ta tiến hành xây dựng mô hình dựa trên config đã chuẩn bị:

```[python3]
# Build dataset
datasets = [build_dataset(cfg.data.train)]

# Build the detector
model = build_detector(cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))
# Add an attribute for visualization convenience
model.CLASSES = ('bicycle', 'motorcycle', 'car', 'van', 'truck', 'bus', 'fire truck')
```

Tiến hành huấn luyện:

```[python3]
train_detector(model, datasets, cfg, distributed=False, validate=True)
```

Log khi huấn luyện:

```
2022-06-20 02:41:11,989 - mmdet - INFO - Start running, host: root@1aa12ebaa861, work_dir: /content/drive/MyDrive/LvThs_OCR/mmdetection/checkpoints
2022-06-20 02:41:11,997 - mmdet - INFO - Hooks will be executed in the following order:
before_run:
(VERY_HIGH   ) StepLrUpdaterHook                  
(NORMAL      ) CheckpointHook                     
(NORMAL      ) EvalHook                           
(VERY_LOW    ) TextLoggerHook                     
 -------------------- 
before_train_epoch:
(VERY_HIGH   ) StepLrUpdaterHook                  
(NORMAL      ) EvalHook                           
(NORMAL      ) NumClassCheckHook                  
(LOW         ) IterTimerHook                      
(VERY_LOW    ) TextLoggerHook                     
 -------------------- 
before_train_iter:
(VERY_HIGH   ) StepLrUpdaterHook                  
(LOW         ) IterTimerHook                      
 -------------------- 
after_train_iter:
(ABOVE_NORMAL) OptimizerHook                      
(NORMAL      ) CheckpointHook                     
(NORMAL      ) EvalHook                           
(LOW         ) IterTimerHook                      
(VERY_LOW    ) TextLoggerHook                     
 -------------------- 
after_train_epoch:
(NORMAL      ) CheckpointHook                     
(NORMAL      ) EvalHook                           
(VERY_LOW    ) TextLoggerHook                     
 -------------------- 
before_val_epoch:
(NORMAL      ) NumClassCheckHook                  
(LOW         ) IterTimerHook                      
(VERY_LOW    ) TextLoggerHook                     
 -------------------- 
before_val_iter:
(LOW         ) IterTimerHook                      
 -------------------- 
after_val_iter:
(LOW         ) IterTimerHook                      
 -------------------- 
after_val_epoch:
(VERY_LOW    ) TextLoggerHook                     
 -------------------- 
2022-06-20 02:41:11,999 - mmdet - INFO - workflow: [('train', 1)], max: 12 epochs
loading annotations into memory...
Done (t=0.01s)
creating index...
index created!
[>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>] 173/173, 8.1 task/s, elapsed: 21s, ETA:     0s2022-06-20 02:44:30,061 - mmdet - INFO - Evaluating bbox...
Loading and preparing results...
DONE (t=0.15s)
creating index...
index created!
Running per image evaluation...
Evaluate annotation type *bbox*
DONE (t=2.42s).
Accumulating evaluation results...
2022-06-20 02:44:33,060 - mmdet - INFO - Epoch(val) [1][327]	bbox_mAP: 0.0060, bbox_mAP_50: 0.0200, bbox_mAP_75: 0.0010, bbox_mAP_s: 0.0090, bbox_mAP_m: 0.0060, bbox_mAP_l: 0.0000, bbox_mAP_copypaste: 0.006 0.020 0.001 0.009 0.006 0.000
DONE (t=0.36s).
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.006
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=1000 ] = 0.020
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=1000 ] = 0.001
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = 0.009
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = 0.006
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.000
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.029
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=300 ] = 0.029
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=1000 ] = 0.029
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = 0.050
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = 0.027
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.000
[>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>] 173/173, 8.0 task/s, elapsed: 21s, ETA:     0s2022-06-20 02:47:51,860 - mmdet - INFO - Evaluating bbox...
Loading and preparing results...
DONE (t=0.15s)
creating index...
index created!
Running per image evaluation...
Evaluate annotation type *bbox*
DONE (t=2.47s).
Accumulating evaluation results...
2022-06-20 02:47:54,962 - mmdet - INFO - Epoch(val) [2][327]	bbox_mAP: 0.0280, bbox_mAP_50: 0.0750, bbox_mAP_75: 0.0150, bbox_mAP_s: 0.0180, bbox_mAP_m: 0.0340, bbox_mAP_l: 0.0140, bbox_mAP_copypaste: 0.028 0.075 0.015 0.018 0.034 0.014
DONE (t=0.40s).
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.028
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=1000 ] = 0.075
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=1000 ] = 0.015
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = 0.018
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = 0.034
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.014
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.092
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=300 ] = 0.092
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=1000 ] = 0.092
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = 0.051
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = 0.071
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.095
2022-06-20 02:50:51,428 - mmdet - INFO - Saving checkpoint at 3 epochs
[>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>] 173/173, 7.9 task/s, elapsed: 22s, ETA:     0s2022-06-20 02:51:15,654 - mmdet - INFO - Evaluating bbox...
Loading and preparing results...
DONE (t=0.02s)
creating index...
index created!
Running per image evaluation...
Evaluate annotation type *bbox*
DONE (t=2.22s).
Accumulating evaluation results...
2022-06-20 02:51:18,523 - mmdet - INFO - Epoch(val) [3][327]	bbox_mAP: 0.0970, bbox_mAP_50: 0.2000, bbox_mAP_75: 0.0800, bbox_mAP_s: 0.0340, bbox_mAP_m: 0.0750, bbox_mAP_l: 0.0900, bbox_mAP_copypaste: 0.097 0.200 0.080 0.034 0.075 0.090
DONE (t=0.42s).
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.097
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=1000 ] = 0.200
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=1000 ] = 0.080
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = 0.034
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = 0.075
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.090
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.158
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=300 ] = 0.158
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=1000 ] = 0.158
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = 0.084
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = 0.117
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.161
[>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>] 173/173, 8.1 task/s, elapsed: 21s, ETA:     0s2022-06-20 02:54:37,466 - mmdet - INFO - Evaluating bbox...
Loading and preparing results...
DONE (t=0.03s)
creating index...
index created!
Running per image evaluation...
Evaluate annotation type *bbox*
DONE (t=2.27s).
Accumulating evaluation results...
2022-06-20 02:54:40,383 - mmdet - INFO - Epoch(val) [4][327]	bbox_mAP: 0.0960, bbox_mAP_50: 0.2050, bbox_mAP_75: 0.0710, bbox_mAP_s: 0.0400, bbox_mAP_m: 0.0910, bbox_mAP_l: 0.0930, bbox_mAP_copypaste: 0.096 0.205 0.071 0.040 0.091 0.093
DONE (t=0.41s).
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.096
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=1000 ] = 0.205
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=1000 ] = 0.071
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = 0.040
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = 0.091
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.093
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.166
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=300 ] = 0.166
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=1000 ] = 0.166
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = 0.092
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = 0.152
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.162
[>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>] 173/173, 8.1 task/s, elapsed: 21s, ETA:     0s2022-06-20 02:57:58,671 - mmdet - INFO - Evaluating bbox...
Loading and preparing results...
DONE (t=0.14s)
creating index...
index created!
Running per image evaluation...
Evaluate annotation type *bbox*
DONE (t=1.36s).
Accumulating evaluation results...
2022-06-20 02:58:00,468 - mmdet - INFO - Epoch(val) [5][327]	bbox_mAP: 0.1600, bbox_mAP_50: 0.3140, bbox_mAP_75: 0.1440, bbox_mAP_s: 0.0670, bbox_mAP_m: 0.1430, bbox_mAP_l: 0.1490, bbox_mAP_copypaste: 0.160 0.314 0.144 0.067 0.143 0.149
DONE (t=0.25s).
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.160
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=1000 ] = 0.314
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=1000 ] = 0.144
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = 0.067
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = 0.143
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.149
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.238
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=300 ] = 0.238
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=1000 ] = 0.238
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = 0.125
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = 0.218
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.232
2022-06-20 03:00:58,010 - mmdet - INFO - Saving checkpoint at 6 epochs
[>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>] 173/173, 7.6 task/s, elapsed: 23s, ETA:     0s2022-06-20 03:01:22,836 - mmdet - INFO - Evaluating bbox...
Loading and preparing results...
DONE (t=0.15s)
creating index...
index created!
Running per image evaluation...
Evaluate annotation type *bbox*
DONE (t=1.80s).
Accumulating evaluation results...
2022-06-20 03:01:25,153 - mmdet - INFO - Epoch(val) [6][327]	bbox_mAP: 0.2180, bbox_mAP_50: 0.4040, bbox_mAP_75: 0.2040, bbox_mAP_s: 0.1180, bbox_mAP_m: 0.2210, bbox_mAP_l: 0.1760, bbox_mAP_copypaste: 0.218 0.404 0.204 0.118 0.221 0.176
DONE (t=0.31s).
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.218
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=1000 ] = 0.404
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=1000 ] = 0.204
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = 0.118
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = 0.221
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.176
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.321
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=300 ] = 0.321
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=1000 ] = 0.321
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = 0.175
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = 0.347
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.261
[>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>] 173/173, 8.1 task/s, elapsed: 21s, ETA:     0s2022-06-20 03:04:44,056 - mmdet - INFO - Evaluating bbox...
Loading and preparing results...
DONE (t=0.02s)
creating index...
index created!
Running per image evaluation...
Evaluate annotation type *bbox*
DONE (t=1.84s).
Accumulating evaluation results...
2022-06-20 03:04:46,289 - mmdet - INFO - Epoch(val) [7][327]	bbox_mAP: 0.2310, bbox_mAP_50: 0.4290, bbox_mAP_75: 0.2300, bbox_mAP_s: 0.1720, bbox_mAP_m: 0.2770, bbox_mAP_l: 0.1790, bbox_mAP_copypaste: 0.231 0.429 0.230 0.172 0.277 0.179
DONE (t=0.31s).
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.231
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=1000 ] = 0.429
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=1000 ] = 0.230
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = 0.172
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = 0.277
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.179
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.369
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=300 ] = 0.369
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=1000 ] = 0.369
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = 0.256
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = 0.407
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.306
[>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>] 173/173, 8.0 task/s, elapsed: 22s, ETA:     0s2022-06-20 03:08:05,146 - mmdet - INFO - Evaluating bbox...
Loading and preparing results...
DONE (t=0.01s)
creating index...
index created!
Running per image evaluation...
Evaluate annotation type *bbox*
DONE (t=1.78s).
Accumulating evaluation results...
2022-06-20 03:08:07,322 - mmdet - INFO - Epoch(val) [8][327]	bbox_mAP: 0.2610, bbox_mAP_50: 0.4790, bbox_mAP_75: 0.2570, bbox_mAP_s: 0.1560, bbox_mAP_m: 0.2850, bbox_mAP_l: 0.2110, bbox_mAP_copypaste: 0.261 0.479 0.257 0.156 0.285 0.211
DONE (t=0.31s).
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.261
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=1000 ] = 0.479
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=1000 ] = 0.257
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = 0.156
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = 0.285
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.211
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.379
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=300 ] = 0.379
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=1000 ] = 0.379
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = 0.223
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = 0.392
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.337
2022-06-20 03:11:04,360 - mmdet - INFO - Saving checkpoint at 9 epochs
[>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>] 173/173, 7.9 task/s, elapsed: 22s, ETA:     0s2022-06-20 03:11:28,443 - mmdet - INFO - Evaluating bbox...
Loading and preparing results...
DONE (t=0.01s)
creating index...
index created!
Running per image evaluation...
Evaluate annotation type *bbox*
DONE (t=1.56s).
Accumulating evaluation results...
2022-06-20 03:11:30,381 - mmdet - INFO - Epoch(val) [9][327]	bbox_mAP: 0.3230, bbox_mAP_50: 0.5630, bbox_mAP_75: 0.3410, bbox_mAP_s: 0.2510, bbox_mAP_m: 0.3380, bbox_mAP_l: 0.2390, bbox_mAP_copypaste: 0.323 0.563 0.341 0.251 0.338 0.239
DONE (t=0.30s).
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.323
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=1000 ] = 0.563
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=1000 ] = 0.341
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = 0.251
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = 0.338
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.239
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.442
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=300 ] = 0.442
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=1000 ] = 0.442
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = 0.365
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = 0.465
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.375
[>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>] 173/173, 8.0 task/s, elapsed: 22s, ETA:     0s2022-06-20 03:14:49,643 - mmdet - INFO - Evaluating bbox...
Loading and preparing results...
DONE (t=0.02s)
creating index...
index created!
Running per image evaluation...
Evaluate annotation type *bbox*
DONE (t=1.56s).
Accumulating evaluation results...
2022-06-20 03:14:51,582 - mmdet - INFO - Epoch(val) [10][327]	bbox_mAP: 0.3260, bbox_mAP_50: 0.5650, bbox_mAP_75: 0.3490, bbox_mAP_s: 0.2750, bbox_mAP_m: 0.3510, bbox_mAP_l: 0.2430, bbox_mAP_copypaste: 0.326 0.565 0.349 0.275 0.351 0.243
DONE (t=0.29s).
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.326
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=1000 ] = 0.565
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=1000 ] = 0.349
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = 0.275
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = 0.351
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.243
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.454
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=300 ] = 0.454
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=1000 ] = 0.454
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = 0.380
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = 0.470
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.394
[>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>] 173/173, 8.0 task/s, elapsed: 22s, ETA:     0s2022-06-20 03:18:11,238 - mmdet - INFO - Evaluating bbox...
Loading and preparing results...
DONE (t=0.14s)
creating index...
index created!
Running per image evaluation...
Evaluate annotation type *bbox*
DONE (t=1.43s).
Accumulating evaluation results...
2022-06-20 03:18:13,126 - mmdet - INFO - Epoch(val) [11][327]	bbox_mAP: 0.3320, bbox_mAP_50: 0.5600, bbox_mAP_75: 0.3570, bbox_mAP_s: 0.3060, bbox_mAP_m: 0.3600, bbox_mAP_l: 0.2460, bbox_mAP_copypaste: 0.332 0.560 0.357 0.306 0.360 0.246
DONE (t=0.27s).
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.332
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=1000 ] = 0.560
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=1000 ] = 0.357
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = 0.306
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = 0.360
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.246
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.457
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=300 ] = 0.457
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=1000 ] = 0.457
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = 0.370
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = 0.477
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.393
```

### Bước 6: Dự đoán trên ảnh mới

Sau khi huấn luyện xong, ta thử dự đoán trên 01 ảnh mới. Giả sử nhóm sinh viên chọn ảnh `000000100.jpg` trong tập dữ liệu:

```[python3]
# Show thử 1 vài ảnh
img = mmcv.imread('/content/drive/MyDrive/BDL_UIT/UIT-VinaDeveS22/images/000000100.jpg')

model.cfg = cfg
result = inference_detector(model, img)
show_result_pyplot(model, img, result)
```

Hình ảnh đầu ra:

![](https://i.imgur.com/jmucRnG.png)

Một số hình ảnh khác:

![](https://i.imgur.com/F9GZ2xY.jpg)

![](https://i.imgur.com/lbUrtUY.png)

![](https://i.imgur.com/M18xAF7.png)

![](https://i.imgur.com/DnELxPD.png)

![](https://i.imgur.com/0MXxOwt.jpg)



## Tài liệu tham khảo

[1]. Girshick, R., Donahue, J., Darrell, T., & Malik, J. (2014). Rich feature hierarchies for accurate object detection and semantic segmentation. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 580-587).

[2]. Girshick, R. (2015). Fast r-cnn. In Proceedings of the IEEE international conference on computer vision (pp. 1440-1448).

[3]. Ren, S., He, K., Girshick, R., & Sun, J. (2015). Faster r-cnn: Towards real-time object detection with region proposal networks. Advances in neural information processing systems, 28.

[4]. Chen, K., Wang, J., Pang, J., Cao, Y., Xiong, Y., Li, X., ... & Lin, D. (2019). MMDetection: Open mmlab detection toolbox and benchmark. arXiv preprint arXiv:1906.07155.
