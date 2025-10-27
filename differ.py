from typing import cast, overload

import paddle
import torch
import atexit
import os
import pandas as pd

DATA_TYPE = torch.Tensor | paddle.Tensor
DISABLE_CMP = True


class Data:
    def __init__(self, data: DATA_TYPE, id: str):
        self.data = data
        self.id = id


class CMP:
    def __init__(
        self,
        name: str,
        eager: bool = False,
        exit_on_error: bool = True,
        drop: bool = False,
        replace: bool = 0,
    ) -> None:
        self.name = name
        self.eager = eager
        self.exit_on_error = exit_on_error
        self.pool: dict[str, list[Data]] = {}
        self.drop = drop
        self.replace = replace
        self.results: dict[str, dict] = {} # for save compare result 

    @overload
    def replace_tensor(self, tensor: torch.Tensor, id: str) -> torch.Tensor: ...
    @overload
    def replace_tensor(self, tensor: paddle.Tensor, id: str) -> paddle.Tensor: ...

    def replace_tensor(self, tensor: DATA_TYPE, id: str):
        if not self.replace:
            return tensor
        device = tensor.device
        if type(tensor) is torch.Tensor:
            for data in self.pool.get(id, []):
                if type(data.data) is paddle.Tensor:
                    if self.drop:
                        self.pool.pop(id)
                    return self.convert_pd_to_pt(data.data).to(device)
            return tensor
        elif type(tensor) is paddle.Tensor:
            for data in self.pool.get(id, []):
                if type(data.data) is torch.Tensor:
                    if self.drop:
                        self.pool.pop(id)
                    return self.convert_pt_to_pd(data.data).to(device)
            return tensor
        else:
            raise TypeError(f"Unsupported tensor type: {type(tensor)}")

    def register(self, tensor: DATA_TYPE, id: str):
        assert tensor.is_contiguous(), f"{id} is not contiguous"
        tensor = tensor.detach()
        data = Data(tensor, id)

        if type(tensor) is torch.Tensor:
            torch.save(tensor, f"save/{self.name}-{id.replace('/', '_')}_pt.pth")
        elif type(tensor) is paddle.Tensor:
            paddle.save(tensor, f"save/{self.name}-{id.replace('/', '_')}_pd")

        if id not in self.pool:
            self.pool[id] = []
        self.pool[id].append(data)
        if self.eager and len(self.pool[id]) > 1:
            self.compare(id)

    def report_error(self, msg):
        print(f"{self.name} - Compare Info:", msg)
        if self.exit_on_error:
            raise RuntimeError(msg)

    def convert_pt_to_pd(self, tensor: torch.Tensor
    ) -> paddle.Tensor:
        if tensor.dtype is torch.bfloat16:
            np_array = tensor.detach().cpu().float().numpy()
            return paddle.tensor(np_array, dtype=paddle.bfloat16)
        else:
            np_array = tensor.detach().cpu().numpy()
            return paddle.tensor(np_array)

    def convert_pd_to_pt(self, tensor: paddle.Tensor) -> torch.Tensor:
        if tensor.dtype == paddle.bfloat16:
            np_array = tensor.cpu().astype("float32").numpy()
            return torch.tensor(np_array, dtype=torch.bfloat16)
        else:
            np_array = tensor.cpu().numpy()
            return torch.tensor(np_array)

    def compare(self, id: str):
        data_list = self.pool[id]
        if len(data_list) > 2:
            raise NotImplementedError(
                "We do not support comparing more than 2 tensors."
            )
        tensor_list = [data.data for data in data_list]
        type_list = [type(tensor) for tensor in tensor_list]
        match type_list:
            case [paddle.Tensor, torch.Tensor]:
                tensor_pd, tensor_pt = tensor_list
            case [torch.Tensor, paddle.Tensor]:
                tensor_pt, tensor_pd = tensor_list
            case _:
                raise NotImplementedError(
                    "We do not support comparing these types of tensors."
                )
        tensor_pd, tensor_pt = (
            cast(paddle.Tensor, tensor_pd),
            cast(torch.Tensor, tensor_pt),
        )
        shape_pd, shape_pt = list(tensor_pd.shape), list(tensor_pt.shape)
        
        result_data = {
            "ID": id,
            "Paddle Shape": str(shape_pd),
            "PyTorch Shape": str(shape_pt),
            "Max Diff": "N/A",
            "Mean Diff": "N/A",
            "Status": ""
        }

        if shape_pd != shape_pt:
            msg = f"{id}'s Shape mismatch: paddle {shape_pd} vs torch {shape_pt}"
            self.report_error(msg)
            result_data["Status"] = "Shape Mismatch"
            self.results[id] = result_data # 存储结果
            return

        tensor_pt_to_pd = self.convert_pt_to_pd(tensor_pt)
        # 确保数据类型一致以进行比较
        if tensor_pd.dtype != tensor_pt_to_pd.dtype:
             tensor_pt_to_pd = tensor_pt_to_pd.astype(tensor_pd.dtype)

        diff = paddle.abs(tensor_pd - tensor_pt_to_pd)
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()
        sum_diff = diff.sum().item()

        msg = f"{id}'s Max diff: {max_diff}, Mean diff: {mean_diff}, Sum diff: {sum_diff}"
        self.report_error(msg)
        
        # 更新结果并存储
        result_data.update({
            "Max Diff": max_diff,
            "Mean Diff": mean_diff,
            "Status": "Compared"
        })
        self.results[id] = result_data

        if self.drop and not self.replace:
            self.pool.pop(id)


class FakeCMP(CMP):
    def __init__(self, name: str) -> None:
        super().__init__(name, eager=False, exit_on_error=False, drop=False, replace=False)

    def register(self, tensor: DATA_TYPE, id: str):
        pass

    def compare(self, id: str):
        pass

    def replace_tensor(self, tensor: DATA_TYPE, id: str):
        return tensor


CMP_POOL: dict[str, CMP] = {}


def get_cmp(
    name: str,
    eager: bool = True,
    exit_on_error: bool = False,
    drop=True,
    replace=0,
) -> CMP:
    if DISABLE_CMP:
        return FakeCMP(name)
    if name not in CMP_POOL:
        CMP_POOL[name] = CMP(
            name=name,
            eager=eager,
            exit_on_error=exit_on_error,
            drop=drop,
            replace=replace,
        )
    return CMP_POOL[name]


def final_check():
    """
    在程序退出时运行，检查所有 CMP 实例。
    如果设置了环境变量 EXPORT_CMP_EXCEL=1，则将比较结果导出到 Excel。
    """
    print("Running final check via atexit...")
    
    # 检查环境变量以决定是否导出 Excel
    # 使用 .lower() in ['1', 'true', 'yes'] 使其更灵活
    should_export = os.getenv('EXPORT_CMP_EXCEL', '0').lower() in ['1', 'true', 'yes']

    if not should_export:
        print("EXPORT_CMP_EXCEL is not set to '1'. Skipping Excel export.")
        # 仍然可以保留旧的未比较项检查逻辑
        for name, cmp_instance in CMP_POOL.items():
            uncompared_ids = [key for key in cmp_instance.pool.keys() if key not in cmp_instance.results]
            if uncompared_ids:
                print(f"Finalizing CMP '{name}', but some tensors are not compared yet:")
                print(uncompared_ids)
        return

    # --- 开始导出逻辑 ---
    output_filename = "comparison_results.xlsx"
    print(f"Exporting comparison results to {output_filename}...")

    # 使用 ExcelWriter 可以在同一个文件中写入多个 sheet
    with pd.ExcelWriter(output_filename, engine='openpyxl') as writer:
        for name, cmp_instance in CMP_POOL.items():
            print(f"Processing sheet for '{name}'...")
            
            report_data = []
            
            # 整合所有已处理和未处理的 ID
            all_ids = set(cmp_instance.pool.keys()) | set(cmp_instance.results.keys())

            for id in sorted(list(all_ids)):
                if id in cmp_instance.results:
                    # 如果 ID 已经比较过，直接使用结果
                    report_data.append(cmp_instance.results[id])
                elif id in cmp_instance.pool:
                    # 如果 ID 在池中但没有结果，说明未比较
                    # (例如，只 register 了一个 tensor)
                    data_list = cmp_instance.pool[id]
                    tensor_info = []
                    for data in data_list:
                        framework = "PyTorch" if isinstance(data.data, torch.Tensor) else "Paddle"
                        tensor_info.append(f"{framework} {list(data.data.shape)}")

                    report_data.append({
                        "ID": id,
                        "Paddle Shape": "N/A",
                        "PyTorch Shape": "N/A",
                        "Max Diff": "N/A",
                        "Mean Diff": "N/A",
                        "Status": f"Uncompared ({', '.join(tensor_info)})"
                    })
            
            if not report_data:
                print(f"  - No data to export for '{name}'.")
                continue

            # 创建 DataFrame
            df = pd.DataFrame(report_data)
            
            # 将 DataFrame 写入到指定名称的 sheet 中
            # index=False 表示不把 DataFrame 的索引写入 Excel
            df.to_excel(writer, sheet_name=name, index=False)
            print(f"  - Sheet '{name}' created with {len(df)} rows.")

    print(f"Successfully exported results to {output_filename}")


# 注册 atexit 函数
atexit.register(final_check)
