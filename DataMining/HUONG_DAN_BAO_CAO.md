# Huong dan bao cao Data Mining

File nay tom tat 3 notebook trong thu muc `DataMining`:

- `K_means.ipynb`: phan cum K-Means tu cai dat.
- `decision_tree.ipynb`: cay quyet dinh ID3 tu cai dat.
- `bayes.ipynb`: Naive Bayes cho bai Play Tennis.

## 1. K-Means

### Muc tieu

K-Means la thuat toan phan cum khong giam sat. Du lieu khong co nhan, minh chon truoc so cum `k`, sau do lap:

1. Gan moi diem vao tam cum gan nhat.
2. Tinh lai tam cum bang trung binh cac diem trong cum.
3. Lap den khi tam cum khong doi nua.

### Du lieu trong code

Code co 12 diem 2 chieu:

- Nhom gan `(1, 1)`: P1-P4.
- Nhom gan `(7, 7)`: P5-P8.
- Nhom gan `(13, 1)`: P9-P12.

Bien quan trong:

```python
data = [
    [1, 1], [1, 2], [2, 1], [2, 2],
    [7, 7], [7, 8], [8, 7], [8, 8],
    [13, 1], [13, 2], [14, 1], [14, 2]
]

k = 3

centroids = [
    data[0].copy(),   # [1, 1]
    data[4].copy(),   # [7, 7]
    data[8].copy()    # [13, 1]
]
```

`k = 3` nghia la chia thanh 3 cum. `centroids` la danh sach tam cum ban dau.

### Giai thich tung ham

#### `euclidean_distance(point, centroid)`

Ham nay tinh khoang cach Euclid giua mot diem va mot tam cum:

```python
total += (point[i] - centroid[i]) ** 2
return math.sqrt(total)
```

Cong thuc:

```text
d = sqrt((x1 - c1)^2 + (x2 - c2)^2 + ...)
```

Neu diem la `[1, 2]`, tam la `[1, 1]` thi:

```text
d = sqrt((1 - 1)^2 + (2 - 1)^2) = 1
```

#### `calculate_centroid(cluster)`

Ham nay tinh tam moi cua mot cum. Tam moi bang trung binh tung cot.

Vi du cum:

```python
[[1, 1], [1, 2], [2, 1], [2, 2]]
```

Tam moi:

```text
x = (1 + 1 + 2 + 2) / 4 = 1.5
y = (1 + 2 + 1 + 2) / 4 = 1.5
=> [1.5, 1.5]
```

Neu cum rong thi tra ve `None`. O phan sau code se giu lai tam cu de tranh loi.

#### `centroids_equal(old_centroids, new_centroids)`

Ham nay kiem tra tam cu va tam moi co gan bang nhau khong.

```python
epsilon = 0.000001
```

Dung `epsilon` vi so thuc co the bi sai so nho. Neu tat ca toa do chenh le <= `epsilon` thi coi nhu khong doi, thuat toan dung.

### Giai thich vong lap chinh

```python
for iteration in range(1, max_iterations + 1):
```

Chay toi da 100 vong de tranh lap vo han.

```python
clusters = [[] for _ in range(k)]
```

Tao `k` cum rong. Neu `k = 3` thi:

```python
[[], [], []]
```

```python
for point_index, point in enumerate(data):
    distances = []
    for centroid in centroids:
        distance = euclidean_distance(point, centroid)
        distances.append(distance)
```

Voi moi diem, tinh khoang cach den tat ca tam cum.

```python
nearest_cluster = distances.index(min(distances))
clusters[nearest_cluster].append(point)
```

Lay cum co khoang cach nho nhat, roi dua diem vao cum do.

Sau khi gan xong tat ca diem:

```python
new_centroid = calculate_centroid(clusters[i])
if new_centroid is None:
    new_centroid = centroids[i]
```

Tinh tam moi. Neu cum rong thi giu tam cu.

Cuoi cung:

```python
if centroids_equal(centroids, new_centroids):
    break
centroids = new_centroids
```

Neu tam moi bang tam cu thi dung. Neu chua bang thi cap nhat tam va lap tiep.

### Ket qua bai hien tai

Vong 1:

- Cum 1: P1, P2, P3, P4 -> tam moi `[1.5, 1.5]`.
- Cum 2: P5, P6, P7, P8 -> tam moi `[7.5, 7.5]`.
- Cum 3: P9, P10, P11, P12 -> tam moi `[13.5, 1.5]`.

Vong 2:

- Cac cum khong doi.
- Tam moi van la `[1.5, 1.5]`, `[7.5, 7.5]`, `[13.5, 1.5]`.
- Thuat toan hoi tu.

### Truong hop thay doi de voi K-Means

#### Thay muon doi so cum

Sua:

```python
k = 3
```

thanh so cum mong muon. Vi du 5 cum:

```python
k = 5
```

Dong thoi phai sua `centroids` co dung 5 tam:

```python
centroids = [
    data[0].copy(),
    data[4].copy(),
    data[8].copy(),
    data[1].copy(),
    data[5].copy()
]
```

Quy tac: `k` bang bao nhieu thi `centroids` phai co bay nhieu tam.

#### Thay bao "5 tam, 3 tam dau lay co dinh, 2 tam cuoi tuy y"

Minh chi can sua phan khoi tao:

```python
k = 5

centroids = [
    data[0].copy(),   # tam 1 co dinh
    data[4].copy(),   # tam 2 co dinh
    data[8].copy(),   # tam 3 co dinh
    data[1].copy(),   # tam 4 tuy y
    data[5].copy()    # tam 5 tuy y
]
```

Neu thay noi "3 tam dau lay 10 cum" kha nang cao thay muon minh biet cach chon tam theo chi so trong `data`. Trong Python chi so bat dau tu 0:

```text
P1  -> data[0]
P2  -> data[1]
P3  -> data[2]
...
P10 -> data[9]
```

Vi du lay 3 tam dau la P1, P5, P10:

```python
centroids = [
    data[0].copy(),   # P1
    data[4].copy(),   # P5
    data[9].copy(),   # P10
    data[2].copy(),   # tuy y
    data[6].copy()    # tuy y
]
```

#### Thay muon nhap tam tu ban phim

Thay khoi `centroids = ...` bang:

```python
k = int(input("Nhap so cum k: "))
centroids = []

for i in range(k):
    x = float(input(f"Nhap x cua tam C{i + 1}: "))
    y = float(input(f"Nhap y cua tam C{i + 1}: "))
    centroids.append([x, y])
```

#### Thay muon chon tam ngau nhien

Them:

```python
import random
```

Va sua:

```python
k = 5
centroids = random.sample(data, k)
```

Can luu y: `k` khong duoc lon hon so diem trong `data` neu dung `random.sample`.

#### Thay them diem du lieu moi

Chi can them vao `data`:

```python
data.append([10, 10])
```

Hoac viet san trong danh sach ban dau. Phan con lai cua code tu chay lai.

#### Thay doi khoang cach

Neu thay muon dung Manhattan thay Euclid, sua ham `euclidean_distance`:

```python
def manhattan_distance(point, centroid):
    total = 0
    for i in range(len(point)):
        total += abs(point[i] - centroid[i])
    return total
```

Va trong code chinh doi:

```python
distance = euclidean_distance(point, centroid)
```

thanh:

```python
distance = manhattan_distance(point, centroid)
```

#### Truong hop co cum rong

Code hien tai da xu ly:

```python
if new_centroid is None:
    new_centroid = centroids[i]
```

Nghia la neu khong co diem nao thuoc cum do thi giu nguyen tam cu. Khi bao cao co the noi: "Em xu ly cum rong bang cach khong cap nhat tam cum do".

## 2. Decision Tree ID3

### Muc tieu

Decision Tree dung de phan lop. Notebook dang dung thuat toan ID3:

1. Tinh Entropy cua tap du lieu.
2. Tinh Information Gain cua tung thuoc tinh.
3. Chon thuoc tinh co Gain cao nhat lam nut chia.
4. Chia du lieu theo gia tri cua thuoc tinh do.
5. Lap de quy den khi tat ca mau cung nhan hoac het thuoc tinh.

### Du lieu trong code

Bai Play Tennis gom 14 mau. Moi mau co:

- `Outlook`: Sunny, Overcast, Rain.
- `Temperature`: Hot, Mild, Cool.
- `Humidity`: High, Normal.
- `Wind`: Weak, Strong.
- `Play`: Yes hoac No.

Ban dau code luu bang tuple:

```python
("Sunny", "Hot", "High", "Weak", "No")
```

Sau do doi sang:

```python
({"Outlook": "Sunny", "Temperature": "Hot", "Humidity": "High", "Wind": "Weak"}, "No")
```

Cach nay giup truy cap thuoc tinh bang ten:

```python
features["Outlook"]
```

### Ham `entropy(samples)`

Entropy do do hon loan cua nhan.

```python
count_yes_no = Counter(label for _, label in samples)
```

Dong nay dem so luong `Yes` va `No`.

Voi tap ban dau:

```text
Yes = 9
No = 5
Entropy(S) = 0.9403
```

Cong thuc:

```text
Entropy(S) = - p_yes * log2(p_yes) - p_no * log2(p_no)
```

Neu entropy = 0 thi tap rat thuan, vi chi co mot nhan. Neu entropy cao thi tap con lan lon.

### Ham `split_data(samples, attribute)`

Ham nay chia du lieu theo mot thuoc tinh.

Vi du chia theo `Outlook`:

- `Sunny`: 5 mau.
- `Overcast`: 4 mau.
- `Rain`: 5 mau.

Ket qua tra ve la dictionary:

```python
{
    "Sunny": [...],
    "Overcast": [...],
    "Rain": [...]
}
```

### Ham `information_gain(samples, attribute)`

Gain cho biet sau khi chia theo mot thuoc tinh thi entropy giam duoc bao nhieu.

```python
entropy_before = entropy(samples)
groups = split_data(samples, attribute)
```

Tinh entropy truoc khi chia, sau do chia nhom.

```python
entropy_after += weight * entropy(group_samples)
```

Entropy sau khi chia la tong entropy cua cac nhom con co nhan voi ti trong.

```python
return entropy_before - entropy_after
```

Gain cang lon thi thuoc tinh chia cang tot.

### Gain o goc cay trong bai hien tai

```text
Gain(Outlook)     = 0.2467
Gain(Temperature) = 0.0292
Gain(Humidity)    = 0.1518
Gain(Wind)        = 0.0481
```

Vay theo ID3 goc cay la `Outlook`, vi `Outlook` co gain cao nhat.

### Ham `best_attribute(samples, remaining_attributes)`

Code hien tai:

```python
best_attr = None
best_gain = -1

for attr in remaining_attributes:
    gain = information_gain(samples, attr)
    if gain > best_gain:
        best_gain = gain
        best_attr = attr
```

Ham duyet tung thuoc tinh, tinh gain, roi lay thuoc tinh co gain lon nhat.

### Class `Node`

Mot node co 2 dang:

1. La la:

```python
node.is_leaf = True
node.label = "Yes"
```

2. La node chia:

```python
node.attribute = "Outlook"
node.children = {
    "Sunny": Node(...),
    "Rain": Node(...)
}
```

### Ham `majority_label(samples)`

Neu het thuoc tinh de chia ma du lieu van con ca Yes va No, code lay nhan xuat hien nhieu nhat:

```python
return Counter(labels).most_common(1)[0][0]
```

### Ham `build_tree(samples, remaining_attributes)`

Day la phan quan trong nhat.

Dieu kien dung 1:

```python
if len(set(labels)) == 1:
```

Neu tat ca mau cung nhan, tao la.

Dieu kien dung 2:

```python
if len(remaining_attributes) == 0:
```

Neu het thuoc tinh de chia, tao la voi nhan da so.

Neu chua dung:

```python
attr, gain = best_attribute(samples, remaining_attributes)
node.attribute = attr
groups = split_data(samples, attr)
next_attributes = [a for a in remaining_attributes if a != attr]
```

Chon thuoc tinh tot nhat, chia du lieu, bo thuoc tinh do khoi danh sach, sau do de quy:

```python
node.children[value] = build_tree(group_samples, next_attributes)
```

### Cay ket qua cua bai hien tai

```text
[Outlook = Sunny]
    [Humidity = High]
        => Play: No
    [Humidity = Normal]
        => Play: Yes
[Outlook = Overcast]
    => Play: Yes
[Outlook = Rain]
    [Wind = Weak]
        => Play: Yes
    [Wind = Strong]
        => Play: No
```

### Cach doc cay khi thay cho vi du

Vi du thay cho:

```text
Outlook = Sunny
Temperature = Cool
Humidity = High
Wind = Strong
```

Doc cay:

1. Goc la `Outlook`.
2. `Outlook = Sunny` thi di vao nhanh Sunny.
3. Node tiep theo la `Humidity`.
4. `Humidity = High` thi ket luan `Play = No`.

Khong can dung `Temperature` va `Wind` neu cay khong hoi den chung tren duong di.

### Truong hop thay doi de voi Decision Tree

#### Thay muon chon goc la gain cao nhat

Code hien tai da lam dung yeu cau nay. Ham `best_attribute` lay `gain > best_gain`.

#### Thay muon chon gain thap nhat

Sua ham `best_attribute`:

```python
def best_attribute(samples, remaining_attributes):
    best_attr = None
    best_gain = float("inf")

    for attr in remaining_attributes:
        gain = information_gain(samples, attr)
        if gain < best_gain:
            best_gain = gain
            best_attr = attr

    return best_attr, best_gain
```

#### Thay muon chon gain o giua

Can sap xep gain, roi lay phan tu giua.

Thay ham `best_attribute` bang:

```python
def best_attribute(samples, remaining_attributes):
    gains = []

    for attr in remaining_attributes:
        gain = information_gain(samples, attr)
        gains.append((gain, attr))

    gains.sort(reverse=True)  # gain cao -> thap

    middle_index = len(gains) // 2
    middle_gain, middle_attr = gains[middle_index]

    return middle_attr, middle_gain
```

Voi 4 thuoc tinh, `len(gains) // 2 = 2`. Sau khi sap xep giam dan:

```text
0: Outlook     0.2467
1: Humidity    0.1518
2: Wind        0.0481
3: Temperature 0.0292
```

Ham tren se chon `Wind` vi no o vi tri index 2. Neu thay noi "gain giua" theo nghia lay trung vi thap hon trong so chan thi dung cach nay.

Neu thay muon lay "gain giua gan trung binh nhat", dung cach sau:

```python
def best_attribute(samples, remaining_attributes):
    gains = []

    for attr in remaining_attributes:
        gain = information_gain(samples, attr)
        gains.append((gain, attr))

    average_gain = sum(gain for gain, _ in gains) / len(gains)
    middle_gain, middle_attr = min(
        gains,
        key=lambda item: abs(item[0] - average_gain)
    )

    return middle_attr, middle_gain
```

Khi thay hoi, nen noi ro: "Neu thay yeu cau gain giua theo sap xep, em sap xep gain roi lay phan tu giua. Neu theo trung binh, em lay gain gan gia tri trung binh nhat".

#### Thay muon ep goc la mot thuoc tinh cu the

Vi du ep goc la `Humidity`, sua trong `build_tree`:

```python
attr, gain = best_attribute(samples, remaining_attributes)
```

thanh:

```python
if len(remaining_attributes) == len(attribute_names):
    attr = "Humidity"
    gain = information_gain(samples, attr)
else:
    attr, gain = best_attribute(samples, remaining_attributes)
```

Y nghia: chi ep o goc cay. Cac node ben duoi van chon theo gain.

#### Thay them cot du lieu moi

Neu them thuoc tinh `Season`, can sua 3 cho:

1. Them gia tri vao moi dong trong `dataset`.
2. Them ten vao `attribute_names`.
3. Them vao dict `features`.

Vi du:

```python
attribute_names = ["Outlook", "Temperature", "Humidity", "Wind", "Season"]
```

Va:

```python
features = {
    "Outlook": row[0],
    "Temperature": row[1],
    "Humidity": row[2],
    "Wind": row[3],
    "Season": row[4],
}
label = row[5]
```

#### Thay cho mau can du doan

Notebook hien tai moi co ham in cay, chua co ham predict. Co the them:

```python
def predict(node, sample):
    if node.is_leaf:
        return node.label

    value = sample[node.attribute]

    if value not in node.children:
        return "Unknown"

    return predict(node.children[value], sample)
```

Dung:

```python
sample = {
    "Outlook": "Sunny",
    "Temperature": "Cool",
    "Humidity": "High",
    "Wind": "Strong"
}

print(predict(tree, sample))
```

#### Truong hop gia tri moi khong co trong cay

Vi du `Outlook = Snow`, trong cay khong co nhanh `Snow`. Ham `predict` ben tren tra ve `"Unknown"`.

Neu muon chuyen ve nhan da so, co the sua thanh:

```python
if value not in node.children:
    return majority_label(data)
```

## 3. Naive Bayes

### Muc tieu

Naive Bayes du doan nhan dua tren xac suat co dieu kien. Code trong `bayes.ipynb` cung dung bai Play Tennis.

Y tuong:

```text
P(yes | features) ti le voi P(yes) * P(outlook|yes) * P(temperature|yes) * P(humidity|yes) * P(wind|yes)
P(no  | features) ti le voi P(no)  * P(outlook|no)  * P(temperature|no)  * P(humidity|no)  * P(wind|no)
```

Chon nhan nao co xac suat lon hon.

### Du lieu trong code

Code tao DataFrame:

```python
df = pd.DataFrame(
    data,
    columns=['outlook', 'temperature', 'humidity', 'wind', 'play']
)
```

Sau do nhap tu ban phim:

```python
outlook = input("Outlook: ")
temperature = input("Temperature: ")
humidity = input("Humidity: ")
wind = input("Wind: ")
```

### Tinh xac suat cho yes

```python
yes_data = df[df['play'] == 'yes']
p_yes = len(yes_data) / len(df)
```

Lay cac dong co `play = yes`, sau do tinh xac suat tien nghiem `P(yes)`.

Voi du lieu hien tai:

```text
P(yes) = 9 / 14
P(no) = 5 / 14
```

Sau do nhan them cac xac suat co dieu kien:

```python
p_yes *= len(yes_data[yes_data['outlook'] == outlook]) / len(yes_data)
p_yes *= len(yes_data[yes_data['temperature'] == temperature]) / len(yes_data)
p_yes *= len(yes_data[yes_data['humidity'] == humidity]) / len(yes_data)
p_yes *= len(yes_data[yes_data['wind'] == wind]) / len(yes_data)
```

Vi du neu `outlook = sunny`, dong dau tien tinh:

```text
P(outlook=sunny | yes)
```

### Tinh xac suat cho no

Tuong tu:

```python
no_data = df[df['play'] == 'no']
p_no = len(no_data) / len(df)
```

Roi nhan:

```python
p_no *= P(outlook | no)
p_no *= P(temperature | no)
p_no *= P(humidity | no)
p_no *= P(wind | no)
```

Cuoi cung:

```python
if p_yes > p_no:
    print("Du doan: YES")
else:
    print("Du doan: NO")
```

### Truong hop thay doi de voi Bayes

#### Thay cho mot mau can du doan

Nhap dung chu thuong theo data:

```text
Outlook: sunny
Temperature: cool
Humidity: high
Wind: strong
```

Code se tinh `P(yes)` va `P(no)`, roi chon cai lon hon.

#### Thay them cot thuoc tinh

Vi du them `season`, can sua:

1. Them cot vao `data`.
2. Them ten cot vao `columns`.
3. Nhap input moi.
4. Nhan them xac suat cho ca yes va no.

Vi du:

```python
season = input("Season: ")

p_yes *= len(yes_data[yes_data['season'] == season]) / len(yes_data)
p_no *= len(no_data[no_data['season'] == season]) / len(no_data)
```

#### Truong hop xac suat bang 0

Code hien tai neu co mot gia tri chua tung xuat hien trong lop `yes` hoac `no` thi xac suat bang 0. Khi do ca tich co the bang 0.

Vi du:

```text
P(outlook=snow | yes) = 0
```

De tranh bang 0, dung Laplace smoothing.

Sua cong thuc dem:

```python
def conditional_probability(df_class, column, value, all_values_count):
    return (len(df_class[df_class[column] == value]) + 1) / (len(df_class) + all_values_count)
```

Dung:

```python
p_yes *= conditional_probability(yes_data, 'outlook', outlook, df['outlook'].nunique())
p_no *= conditional_probability(no_data, 'outlook', outlook, df['outlook'].nunique())
```

Lam tuong tu cho cac cot khac.

#### Thay muon khong nhap input ma viet san mau

Thay:

```python
outlook = input("Outlook: ")
temperature = input("Temperature: ")
humidity = input("Humidity: ")
wind = input("Wind: ")
```

bang:

```python
outlook = "sunny"
temperature = "cool"
humidity = "high"
wind = "strong"
```

#### Thay muon in ket qua dang phan tram

Them:

```python
total = p_yes + p_no
print("P(yes) normalized =", p_yes / total)
print("P(no) normalized =", p_no / total)
```

Can luu y: `p_yes` va `p_no` ban dau la diem so ti le, chua phai xac suat chuan hoa tong bang 1.

## 4. Cau tra loi nhanh khi thay hoi

### K-Means

- `k` la so cum can chia.
- `centroids` la cac tam cum ban dau.
- Moi vong lap gom 2 buoc: gan diem vao tam gan nhat, tinh lai tam.
- Khoang cach dang dung la Euclid.
- Dung khi tam moi khong doi so voi tam cu.
- Neu cum rong, code giu lai tam cu.
- Muon doi so cum thi doi `k` va so phan tu trong `centroids`.

### Decision Tree

- Entropy do do hon loan cua tap nhan.
- Information Gain do muc giam entropy sau khi chia theo thuoc tinh.
- ID3 chon thuoc tinh co gain cao nhat.
- Goc cay hien tai la `Outlook`.
- Neu tat ca mau cung nhan thi tao node la.
- Neu het thuoc tinh thi lay nhan da so.
- Muon chon gain giua thi sua ham `best_attribute`.

### Bayes

- `P(yes)` va `P(no)` la xac suat tien nghiem.
- Sau do nhan voi cac xac suat co dieu kien cua tung thuoc tinh.
- Chon nhan co gia tri lon hon.
- Code hien tai chua dung Laplace smoothing, nen co the bi xac suat 0 neu gap gia tri moi.
- Muon them thuoc tinh thi them cot, them input, va nhan them xac suat co dieu kien.
