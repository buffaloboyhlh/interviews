# PyMilvus 使用教程：从入门到精通

## 1. Milvus 简介

Milvus 是一个开源的向量数据库，专门用于处理大规模向量数据的存储和相似度搜索。它支持多种索引类型和距离度量方式，广泛应用于推荐系统、图像检索、自然语言处理等领域。

### 核心特性：
- 高性能向量相似度搜索
- 支持多种索引类型（IVF_FLAT、HNSW、ANNOY等）
- 可扩展的分布式架构
- 支持多种距离度量（L2、内积、余弦等）
- 丰富的 SDK 支持

## 2. 环境准备

### 安装 PyMilvus

```bash
pip install pymilvus
```

### 启动 Milvus 服务

使用 Docker 启动 Milvus 单机版：

```bash
docker run -d --name milvus-standalone \
  -p 19530:19530 \
  -p 9091:9091 \
  milvusdb/milvus:v2.3.4
```

## 3. 基础概念

### 核心组件

1. **Collection（集合）**：类似关系数据库中的表，包含多个实体
2. **Field（字段）**：集合中的列，可以是标量或向量
3. **Entity（实体）**：集合中的一行数据
4. **Partition（分区）**：用于数据管理的逻辑分组
5. **Index（索引）**：加速向量搜索的数据结构
6. **Schema（模式）**：定义集合的结构

### 距离度量

- **L2（欧氏距离）**：`"L2"`
- **内积**：`"IP"`
- **余弦相似度**：`"COSINE"`

## 4. 基础操作

### 连接 Milvus

```python
from pymilvus import connections, utility

# 连接 Milvus
connections.connect(
    alias="default",
    host="localhost",
    port="19530"
)

# 检查连接状态
print(f"Milvus 版本: {utility.get_server_version()}")
```

### 创建集合

```python
from pymilvus import Collection, FieldSchema, CollectionSchema, DataType

# 定义字段
fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=128),
    FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=200),
    FieldSchema(name="category", dtype=DataType.VARCHAR, max_length=100)
]

# 创建模式
schema = CollectionSchema(
    fields=fields,
    description="示例集合",
    enable_dynamic_field=True
)

# 创建集合
collection_name = "example_collection"
collection = Collection(
    name=collection_name,
    schema=schema,
    using="default"
)

print(f"集合 {collection_name} 创建成功")
```

### 插入数据

```python
import random

# 生成示例数据
num_entities = 1000
embeddings = [[random.random() for _ in range(128)] for _ in range(num_entities)]
titles = [f"标题_{i}" for i in range(num_entities)]
categories = [f"分类_{random.randint(1, 5)}" for _ in range(num_entities)]

# 准备插入数据
data = [
    embeddings,  # 向量数据
    titles,      # 标量数据
    categories   # 标量数据
]

# 插入数据
insert_result = collection.insert(data)

print(f"插入了 {len(insert_result.primary_keys)} 条数据")
print(f"主键: {insert_result.primary_keys[:5]}")  # 显示前5个主键
```

### 创建索引

```python
# 在插入数据后创建索引
index_params = {
    "index_type": "IVF_FLAT",
    "metric_type": "L2",
    "params": {"nlist": 128}
}

# 为向量字段创建索引
collection.create_index(
    field_name="embedding",
    index_params=index_params
)

print("索引创建成功")
```

### 搜索向量

```python
# 加载集合到内存
collection.load()

# 准备搜索向量
search_vectors = [[random.random() for _ in range(128)] for _ in range(5)]

# 设置搜索参数
search_params = {
    "metric_type": "L2",
    "params": {"nprobe": 10}
}

# 执行搜索
results = collection.search(
    data=search_vectors,
    anns_field="embedding",
    param=search_params,
    limit=3,  # 返回top-3结果
    output_fields=["title", "category"]  # 返回的字段
)

# 解析搜索结果
for i, result in enumerate(results):
    print(f"\n搜索向量 {i+1} 的结果:")
    for j, hit in enumerate(result):
        print(f"  第 {j+1} 名: ID={hit.id}, 距离={hit.distance:.4f}, "
              f"标题={hit.entity.get('title')}, 分类={hit.entity.get('category')}")
```

### 查询标量数据

```python
# 基于标量字段查询
query_result = collection.query(
    expr='category == "分类_1"',
    output_fields=["id", "title", "category"],
    limit=5
)

print("\n分类为 '分类_1' 的数据:")
for result in query_result:
    print(f"ID: {result['id']}, 标题: {result['title']}, 分类: {result['category']}")
```

## 5. 进阶功能

### 分区管理

```python
# 创建分区
partition_name = "partition_1"
collection.create_partition(partition_name)

# 向分区插入数据
partition_data = [
    [[random.random() for _ in range(128)] for _ in range(100)],
    [f"分区标题_{i}" for i in range(100)],
    ["特殊分类" for _ in range(100)]
]

partition_insert_result = collection.insert(partition_data, partition_name=partition_name)

# 在指定分区搜索
partition_results = collection.search(
    data=search_vectors,
    anns_field="embedding",
    param=search_params,
    limit=2,
    partition_names=[partition_name],
    output_fields=["title", "category"]
)
```

### 混合搜索（标量 + 向量）

```python
# 结合标量过滤的向量搜索
hybrid_search_results = collection.search(
    data=search_vectors,
    anns_field="embedding",
    param=search_params,
    limit=5,
    expr='category == "分类_2"',  # 标量过滤条件
    output_fields=["title", "category"]
)

print("\n混合搜索结果 (分类为 '分类_2'):")
for i, result in enumerate(hybrid_search_results):
    print(f"搜索向量 {i+1}:")
    for hit in result:
        print(f"  ID={hit.id}, 距离={hit.distance:.4f}, "
              f"标题={hit.entity.get('title')}")
```

### 数据删除

```python
# 删除特定数据
delete_expr = 'category == "特殊分类"'
collection.delete(delete_expr)

print("数据删除完成")
```

### 集合管理

```python
# 获取集合信息
collection_info = {
    "实体数量": collection.num_entities,
    "分区列表": collection.partitions
}

print("集合信息:")
for key, value in collection_info.items():
    print(f"  {key}: {value}")

# 释放集合内存
collection.release()
```

## 6. 性能优化

### 索引选择策略

```python
def create_optimized_index(collection, field_name, data_size):
    """根据数据量选择最优索引"""
    if data_size < 10000:
        # 小数据量使用FLAT
        index_params = {
            "index_type": "FLAT",
            "metric_type": "L2"
        }
    elif data_size < 1000000:
        # 中等数据量使用IVF_FLAT
        index_params = {
            "index_type": "IVF_FLAT",
            "metric_type": "L2",
            "params": {"nlist": 1024}
        }
    else:
        # 大数据量使用HNSW
        index_params = {
            "index_type": "HNSW",
            "metric_type": "L2",
            "params": {"M": 16, "efConstruction": 200}
        }
    
    collection.create_index(field_name, index_params)
    return index_params

# 使用优化索引
optimized_index = create_optimized_index(collection, "embedding", 100000)
print(f"创建的索引参数: {optimized_index}")
```

### 批量操作优化

```python
def batch_insert(collection, data, batch_size=1000):
    """批量插入数据优化"""
    total_records = len(data[0])
    inserted_count = 0
    
    for i in range(0, total_records, batch_size):
        end_idx = min(i + batch_size, total_records)
        batch_data = [field[i:end_idx] for field in data]
        
        insert_result = collection.insert(batch_data)
        inserted_count += len(insert_result.primary_keys)
        
        print(f"已插入 {inserted_count}/{total_records} 条数据")
    
    return inserted_count

# 使用批量插入
large_embeddings = [[random.random() for _ in range(128)] for _ in range(5000)]
large_titles = [f"批量标题_{i}" for i in range(5000)]
large_categories = [f"分类_{random.randint(1, 3)}" for _ in range(5000)]

large_data = [large_embeddings, large_titles, large_categories]
batch_insert(collection, large_data, batch_size=500)
```

## 7. 实战案例

### 案例：图像检索系统

```python
import numpy as np
from PIL import Image
import requests
from io import BytesIO

class ImageSearchSystem:
    def __init__(self, collection_name="image_search"):
        self.collection_name = collection_name
        self.setup_collection()
    
    def setup_collection(self):
        """设置图像搜索集合"""
        if utility.has_collection(self.collection_name):
            self.collection = Collection(self.collection_name)
            return
        
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="image_vector", dtype=DataType.FLOAT_VECTOR, dim=512),
            FieldSchema(name="image_url", dtype=DataType.VARCHAR, max_length=500),
            FieldSchema(name="tags", dtype=DataType.VARCHAR, max_length=200),
            FieldSchema(name="upload_time", dtype=DataType.INT64)
        ]
        
        schema = CollectionSchema(fields, "图像搜索集合")
        self.collection = Collection(self.collection_name, schema)
        
        # 创建索引
        index_params = {
            "index_type": "IVF_FLAT",
            "metric_type": "COSINE",  # 图像搜索通常使用余弦相似度
            "params": {"nlist": 1024}
        }
        self.collection.create_index("image_vector", index_params)
    
    def add_image(self, image_vector, image_url, tags):
        """添加图像到系统"""
        current_time = int(time.time())
        data = [
            [image_vector],
            [image_url],
            [tags],
            [current_time]
        ]
        
        result = self.collection.insert(data)
        return result.primary_keys[0]
    
    def search_similar_images(self, query_vector, top_k=10, tag_filter=None):
        """搜索相似图像"""
        self.collection.load()
        
        search_params = {
            "metric_type": "COSINE",
            "params": {"nprobe": 16}
        }
        
        expr = None
        if tag_filter:
            expr = f'tags like "%{tag_filter}%"'
        
        results = self.collection.search(
            data=[query_vector],
            anns_field="image_vector",
            param=search_params,
            limit=top_k,
            expr=expr,
            output_fields=["image_url", "tags", "upload_time"]
        )
        
        return results[0]
    
    def demo_simulation(self):
        """演示模拟"""
        print("=== 图像检索系统演示 ===")
        
        # 模拟添加图像数据
        print("\n1. 添加图像数据...")
        for i in range(100):
            # 模拟512维图像特征向量
            fake_vector = [random.random() for _ in range(512)]
            tags = f"tag_{random.randint(1, 10)}"
            self.add_image(fake_vector, f"http://example.com/image_{i}.jpg", tags)
        
        # 模拟搜索
        print("\n2. 执行相似图像搜索...")
        query_vector = [random.random() for _ in range(512)]
        results = self.search_similar_images(query_vector, top_k=5)
        
        print("搜索结果:")
        for i, hit in enumerate(results):
            print(f"  {i+1}. 相似度: {1 - hit.distance:.4f}, "
                  f"URL: {hit.entity.get('image_url')}, "
                  f"标签: {hit.entity.get('tags')}")

# 运行演示
image_system = ImageSearchSystem()
image_system.demo_simulation()
```

### 案例：推荐系统

```python
class RecommendationSystem:
    def __init__(self):
        self.collection_name = "user_embeddings"
        self.setup_collection()
    
    def setup_collection(self):
        """设置用户嵌入向量集合"""
        if utility.has_collection(self.collection_name):
            self.collection = Collection(self.collection_name)
            return
        
        fields = [
            FieldSchema(name="user_id", dtype=DataType.INT64, is_primary=True),
            FieldSchema(name="user_embedding", dtype=DataType.FLOAT_VECTOR, dim=256),
            FieldSchema(name="user_features", dtype=DataType.VARCHAR, max_length=500),
            FieldSchema(name="last_active", dtype=DataType.INT64)
        ]
        
        schema = CollectionSchema(fields, "用户推荐系统")
        self.collection = Collection(self.collection_name, schema)
        
        index_params = {
            "index_type": "HNSW",
            "metric_type": "IP",  # 推荐系统常用内积
            "params": {"M": 16, "efConstruction": 200}
        }
        self.collection.create_index("user_embedding", index_params)
    
    def add_user(self, user_id, embedding, features):
        """添加用户"""
        current_time = int(time.time())
        data = [
            [user_id],
            [embedding],
            [features],
            [current_time]
        ]
        
        self.collection.insert(data)
    
    def find_similar_users(self, user_id, top_k=5):
        """查找相似用户"""
        self.collection.load()
        
        # 先获取目标用户的向量
        target_user = self.collection.query(
            expr=f"user_id == {user_id}",
            output_fields=["user_embedding"]
        )
        
        if not target_user:
            return []
        
        target_embedding = target_user[0]["user_embedding"]
        
        search_params = {
            "metric_type": "IP",
            "params": {"ef": 50}
        }
        
        results = self.collection.search(
            data=[target_embedding],
            anns_field="user_embedding",
            param=search_params,
            limit=top_k + 1,  # 包含自己
            output_fields=["user_id", "user_features"]
        )
        
        # 过滤掉自己
        similar_users = []
        for hit in results[0]:
            if hit.id != user_id:
                similar_users.append({
                    "user_id": hit.id,
                    "score": hit.distance,
                    "features": hit.entity.get("user_features")
                })
        
        return similar_users[:top_k]

# 使用示例
rec_system = RecommendationSystem()

# 添加示例用户
for i in range(1, 101):
    embedding = [random.random() for _ in range(256)]
    features = f"age_{random.randint(18, 60)},interests_{random.randint(1, 10)}"
    rec_system.add_user(i, embedding, features)

# 查找相似用户
similar_users = rec_system.find_similar_users(1, top_k=3)
print("相似用户推荐:")
for user in similar_users:
    print(f"用户ID: {user['user_id']}, 相似度: {user['score']:.4f}")
```

## 总结

本教程涵盖了 PyMilvus 从基础到进阶的完整使用流程：

1. **基础概念**：理解 Milvus 的核心组件和数据模型
2. **基础操作**：连接、集合管理、数据插入、搜索查询
3. **进阶功能**：分区管理、混合搜索、数据删除
4. **性能优化**：索引选择、批量操作
5. **实战案例**：图像检索、推荐系统等实际应用

通过掌握这些知识，你可以在实际项目中高效地使用 Milvus 进行向量数据的管理和相似度搜索。记得根据具体业务场景选择合适的索引类型和搜索参数，以达到最佳的性能效果。