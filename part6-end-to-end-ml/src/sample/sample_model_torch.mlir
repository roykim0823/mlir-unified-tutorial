module {
  func.func @sample_model(%arg0: !torch.vtensor<[3,4],f32>) -> !torch.vtensor<[3,5],f32> {
    %int0 = torch.constant.int 0
    %float1.000000e00 = torch.constant.float 1.000000e+00
    %float0.000000e00 = torch.constant.float 0.000000e+00
    %0 = torch.vtensor.literal(dense_resource<torch_tensor_5_torch.float32> : tensor<5xf32>) : !torch.vtensor<[5],f32>
    %1 = torch.vtensor.literal(dense_resource<torch_tensor_5_4_torch.float32> : tensor<5x4xf32>) : !torch.vtensor<[5,4],f32>
    %2 = torch.vtensor.literal(dense_resource<torch_tensor_3_4_torch.float32> : tensor<3x4xf32>) : !torch.vtensor<[3,4],f32>
    %int1 = torch.constant.int 1
    %3 = torch.aten.add.Tensor %arg0, %2, %int1 : !torch.vtensor<[3,4],f32>, !torch.vtensor<[3,4],f32>, !torch.int -> !torch.vtensor<[3,4],f32>
    %4 = torch.aten.transpose.int %1, %int0, %int1 : !torch.vtensor<[5,4],f32>, !torch.int, !torch.int -> !torch.vtensor<[4,5],f32>
    %5 = torch.aten.mm %3, %4 : !torch.vtensor<[3,4],f32>, !torch.vtensor<[4,5],f32> -> !torch.vtensor<[3,5],f32>
    %6 = torch.aten.add.Tensor %5, %0, %int1 : !torch.vtensor<[3,5],f32>, !torch.vtensor<[5],f32>, !torch.int -> !torch.vtensor<[3,5],f32>
    %7 = torch.aten.clamp %6, %float0.000000e00, %float1.000000e00 : !torch.vtensor<[3,5],f32>, !torch.float, !torch.float -> !torch.vtensor<[3,5],f32>
    return %7 : !torch.vtensor<[3,5],f32>
  }
}

{-#
  dialect_resources: {
    builtin: {
      torch_tensor_5_torch.float32: "0x040000006061C8BC30FC8ABD20AE6A3DA27BF63E38A6953E",
      torch_tensor_5_4_torch.float32: "0x04000000D8A9C13E809B263DFA4EB8BE4635EC3E64B9BB3E80CE3DBC20A895BEB04E46BE5409E2BE7AA7A93E70B931BD6661B03E70C7403E4025CBBD6AB2A3BE7031323E9842A4BE4892E3BDEC2F983EDE668BBE",
      torch_tensor_3_4_torch.float32: "0x04000000001F723E8C0E683E8CEF4C3F883F2D3EA0A8873E41A1453FCC43033E59C63E3F97F14D3FD5BD223F1AF1163F6D7B313F"
    }
  }
#-}
