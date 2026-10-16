#map = affine_map<(d0, d1, d2) -> (d0, d1)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map2 = affine_map<(d0) -> (d0)>
#map3 = affine_map<(d0, d1) -> (d0, d1)>
#map4 = affine_map<() -> ()>
#map5 = affine_map<(d0, d1, d2, d3) -> (d0, d1, 0, d3)>
#map6 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>
#map7 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map8 = affine_map<(d0, d1, d2, d3) -> ()>
#map9 = affine_map<(d0, d1, d2, d3) -> (0, 0, 0, 0)>
#map10 = affine_map<(d0, d1, d2, d3) -> (0, 0, 0, d3)>
#map11 = affine_map<(d0, d1, d2) -> (d0, d1, 0)>
#map12 = affine_map<(d0, d1, d2) -> (d2)>
#map13 = affine_map<(d0, d1) -> (d1)>
#map14 = affine_map<(d0, d1, d2, d3) -> (d0, 0, d2, d3)>
#map15 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
module {
  func.func @gpt2_model(%arg0: tensor<1x?xi64>, %arg1: tensor<1x?xi64>) -> tensor<1x?x50257xf32> {
    %c768_i64 = arith.constant 768 : i64
    %c3_i64 = arith.constant 3 : i64
    %c0_i64 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c50257 = arith.constant 50257 : index
    %c1024 = arith.constant 1024 : index
    %c3 = arith.constant 3 : index
    %cst = arith.constant 0.000000e+00 : f64
    %cst_0 = arith.constant 0.000000e+00 : f32
    %c64_i64 = arith.constant 64 : i64
    %c12_i64 = arith.constant 12 : i64
    %cst_1 = arith.constant 0xFF800000 : f32
    %false = arith.constant false
    %cst_2 = arith.constant dense<0.000000e+00> : tensor<f32>
    %cst_3 = arith.constant dense<0xFF800000> : tensor<f32>
    %cst_4 = arith.constant dense_resource<torch_tensor_768_torch.float32_73> : tensor<768xf32>
    %cst_5 = arith.constant dense_resource<torch_tensor_768_torch.float32_72> : tensor<768xf32>
    %cst_6 = arith.constant dense_resource<torch_tensor_768_torch.float32_71> : tensor<768xf32>
    %cst_7 = arith.constant dense_resource<torch_tensor_3072_768_torch.float32_11> : tensor<3072x768xf32>
    %cst_8 = arith.constant dense_resource<torch_tensor_3072_torch.float32_11> : tensor<3072xf32>
    %cst_9 = arith.constant dense_resource<torch_tensor_768_3072_torch.float32_11> : tensor<768x3072xf32>
    %cst_10 = arith.constant dense_resource<torch_tensor_768_torch.float32_70> : tensor<768xf32>
    %cst_11 = arith.constant dense_resource<torch_tensor_768_torch.float32_69> : tensor<768xf32>
    %cst_12 = arith.constant dense_resource<torch_tensor_768_torch.float32_68> : tensor<768xf32>
    %cst_13 = arith.constant dense_resource<torch_tensor_768_768_torch.float32_11> : tensor<768x768xf32>
    %cst_14 = arith.constant dense_resource<torch_tensor_2304_torch.float32_11> : tensor<2304xf32>
    %cst_15 = arith.constant dense_resource<torch_tensor_768_2304_torch.float32_11> : tensor<768x2304xf32>
    %cst_16 = arith.constant dense_resource<torch_tensor_768_torch.float32_67> : tensor<768xf32>
    %cst_17 = arith.constant dense_resource<torch_tensor_768_torch.float32_66> : tensor<768xf32>
    %cst_18 = arith.constant dense_resource<torch_tensor_768_torch.float32_65> : tensor<768xf32>
    %cst_19 = arith.constant dense_resource<torch_tensor_3072_768_torch.float32_10> : tensor<3072x768xf32>
    %cst_20 = arith.constant dense_resource<torch_tensor_3072_torch.float32_10> : tensor<3072xf32>
    %cst_21 = arith.constant dense_resource<torch_tensor_768_3072_torch.float32_10> : tensor<768x3072xf32>
    %cst_22 = arith.constant dense_resource<torch_tensor_768_torch.float32_64> : tensor<768xf32>
    %cst_23 = arith.constant dense_resource<torch_tensor_768_torch.float32_63> : tensor<768xf32>
    %cst_24 = arith.constant dense_resource<torch_tensor_768_torch.float32_62> : tensor<768xf32>
    %cst_25 = arith.constant dense_resource<torch_tensor_768_768_torch.float32_10> : tensor<768x768xf32>
    %cst_26 = arith.constant dense_resource<torch_tensor_2304_torch.float32_10> : tensor<2304xf32>
    %cst_27 = arith.constant dense_resource<torch_tensor_768_2304_torch.float32_10> : tensor<768x2304xf32>
    %cst_28 = arith.constant dense_resource<torch_tensor_768_torch.float32_61> : tensor<768xf32>
    %cst_29 = arith.constant dense_resource<torch_tensor_768_torch.float32_60> : tensor<768xf32>
    %cst_30 = arith.constant dense_resource<torch_tensor_768_torch.float32_59> : tensor<768xf32>
    %cst_31 = arith.constant dense_resource<torch_tensor_3072_768_torch.float32_9> : tensor<3072x768xf32>
    %cst_32 = arith.constant dense_resource<torch_tensor_3072_torch.float32_9> : tensor<3072xf32>
    %cst_33 = arith.constant dense_resource<torch_tensor_768_3072_torch.float32_9> : tensor<768x3072xf32>
    %cst_34 = arith.constant dense_resource<torch_tensor_768_torch.float32_58> : tensor<768xf32>
    %cst_35 = arith.constant dense_resource<torch_tensor_768_torch.float32_57> : tensor<768xf32>
    %cst_36 = arith.constant dense_resource<torch_tensor_768_torch.float32_56> : tensor<768xf32>
    %cst_37 = arith.constant dense_resource<torch_tensor_768_768_torch.float32_9> : tensor<768x768xf32>
    %cst_38 = arith.constant dense_resource<torch_tensor_2304_torch.float32_9> : tensor<2304xf32>
    %cst_39 = arith.constant dense_resource<torch_tensor_768_2304_torch.float32_9> : tensor<768x2304xf32>
    %cst_40 = arith.constant dense_resource<torch_tensor_768_torch.float32_55> : tensor<768xf32>
    %cst_41 = arith.constant dense_resource<torch_tensor_768_torch.float32_54> : tensor<768xf32>
    %cst_42 = arith.constant dense_resource<torch_tensor_768_torch.float32_53> : tensor<768xf32>
    %cst_43 = arith.constant dense_resource<torch_tensor_3072_768_torch.float32_8> : tensor<3072x768xf32>
    %cst_44 = arith.constant dense_resource<torch_tensor_3072_torch.float32_8> : tensor<3072xf32>
    %cst_45 = arith.constant dense_resource<torch_tensor_768_3072_torch.float32_8> : tensor<768x3072xf32>
    %cst_46 = arith.constant dense_resource<torch_tensor_768_torch.float32_52> : tensor<768xf32>
    %cst_47 = arith.constant dense_resource<torch_tensor_768_torch.float32_51> : tensor<768xf32>
    %cst_48 = arith.constant dense_resource<torch_tensor_768_torch.float32_50> : tensor<768xf32>
    %cst_49 = arith.constant dense_resource<torch_tensor_768_768_torch.float32_8> : tensor<768x768xf32>
    %cst_50 = arith.constant dense_resource<torch_tensor_2304_torch.float32_8> : tensor<2304xf32>
    %cst_51 = arith.constant dense_resource<torch_tensor_768_2304_torch.float32_8> : tensor<768x2304xf32>
    %cst_52 = arith.constant dense_resource<torch_tensor_768_torch.float32_49> : tensor<768xf32>
    %cst_53 = arith.constant dense_resource<torch_tensor_768_torch.float32_48> : tensor<768xf32>
    %cst_54 = arith.constant dense_resource<torch_tensor_768_torch.float32_47> : tensor<768xf32>
    %cst_55 = arith.constant dense_resource<torch_tensor_3072_768_torch.float32_7> : tensor<3072x768xf32>
    %cst_56 = arith.constant dense_resource<torch_tensor_3072_torch.float32_7> : tensor<3072xf32>
    %cst_57 = arith.constant dense_resource<torch_tensor_768_3072_torch.float32_7> : tensor<768x3072xf32>
    %cst_58 = arith.constant dense_resource<torch_tensor_768_torch.float32_46> : tensor<768xf32>
    %cst_59 = arith.constant dense_resource<torch_tensor_768_torch.float32_45> : tensor<768xf32>
    %cst_60 = arith.constant dense_resource<torch_tensor_768_torch.float32_44> : tensor<768xf32>
    %cst_61 = arith.constant dense_resource<torch_tensor_768_768_torch.float32_7> : tensor<768x768xf32>
    %cst_62 = arith.constant dense_resource<torch_tensor_2304_torch.float32_7> : tensor<2304xf32>
    %cst_63 = arith.constant dense_resource<torch_tensor_768_2304_torch.float32_7> : tensor<768x2304xf32>
    %cst_64 = arith.constant dense_resource<torch_tensor_768_torch.float32_43> : tensor<768xf32>
    %cst_65 = arith.constant dense_resource<torch_tensor_768_torch.float32_42> : tensor<768xf32>
    %cst_66 = arith.constant dense_resource<torch_tensor_768_torch.float32_41> : tensor<768xf32>
    %cst_67 = arith.constant dense_resource<torch_tensor_3072_768_torch.float32_6> : tensor<3072x768xf32>
    %cst_68 = arith.constant dense_resource<torch_tensor_3072_torch.float32_6> : tensor<3072xf32>
    %cst_69 = arith.constant dense_resource<torch_tensor_768_3072_torch.float32_6> : tensor<768x3072xf32>
    %cst_70 = arith.constant dense_resource<torch_tensor_768_torch.float32_40> : tensor<768xf32>
    %cst_71 = arith.constant dense_resource<torch_tensor_768_torch.float32_39> : tensor<768xf32>
    %cst_72 = arith.constant dense_resource<torch_tensor_768_torch.float32_38> : tensor<768xf32>
    %cst_73 = arith.constant dense_resource<torch_tensor_768_768_torch.float32_6> : tensor<768x768xf32>
    %cst_74 = arith.constant dense_resource<torch_tensor_2304_torch.float32_6> : tensor<2304xf32>
    %cst_75 = arith.constant dense_resource<torch_tensor_768_2304_torch.float32_6> : tensor<768x2304xf32>
    %cst_76 = arith.constant dense_resource<torch_tensor_768_torch.float32_37> : tensor<768xf32>
    %cst_77 = arith.constant dense_resource<torch_tensor_768_torch.float32_36> : tensor<768xf32>
    %cst_78 = arith.constant dense_resource<torch_tensor_768_torch.float32_35> : tensor<768xf32>
    %cst_79 = arith.constant dense_resource<torch_tensor_3072_768_torch.float32_5> : tensor<3072x768xf32>
    %cst_80 = arith.constant dense_resource<torch_tensor_3072_torch.float32_5> : tensor<3072xf32>
    %cst_81 = arith.constant dense_resource<torch_tensor_768_3072_torch.float32_5> : tensor<768x3072xf32>
    %cst_82 = arith.constant dense_resource<torch_tensor_768_torch.float32_34> : tensor<768xf32>
    %cst_83 = arith.constant dense_resource<torch_tensor_768_torch.float32_33> : tensor<768xf32>
    %cst_84 = arith.constant dense_resource<torch_tensor_768_torch.float32_32> : tensor<768xf32>
    %cst_85 = arith.constant dense_resource<torch_tensor_768_768_torch.float32_5> : tensor<768x768xf32>
    %cst_86 = arith.constant dense_resource<torch_tensor_2304_torch.float32_5> : tensor<2304xf32>
    %cst_87 = arith.constant dense_resource<torch_tensor_768_2304_torch.float32_5> : tensor<768x2304xf32>
    %cst_88 = arith.constant dense_resource<torch_tensor_768_torch.float32_31> : tensor<768xf32>
    %cst_89 = arith.constant dense_resource<torch_tensor_768_torch.float32_30> : tensor<768xf32>
    %cst_90 = arith.constant dense_resource<torch_tensor_768_torch.float32_29> : tensor<768xf32>
    %cst_91 = arith.constant dense_resource<torch_tensor_3072_768_torch.float32_4> : tensor<3072x768xf32>
    %cst_92 = arith.constant dense_resource<torch_tensor_3072_torch.float32_4> : tensor<3072xf32>
    %cst_93 = arith.constant dense_resource<torch_tensor_768_3072_torch.float32_4> : tensor<768x3072xf32>
    %cst_94 = arith.constant dense_resource<torch_tensor_768_torch.float32_28> : tensor<768xf32>
    %cst_95 = arith.constant dense_resource<torch_tensor_768_torch.float32_27> : tensor<768xf32>
    %cst_96 = arith.constant dense_resource<torch_tensor_768_torch.float32_26> : tensor<768xf32>
    %cst_97 = arith.constant dense_resource<torch_tensor_768_768_torch.float32_4> : tensor<768x768xf32>
    %cst_98 = arith.constant dense_resource<torch_tensor_2304_torch.float32_4> : tensor<2304xf32>
    %cst_99 = arith.constant dense_resource<torch_tensor_768_2304_torch.float32_4> : tensor<768x2304xf32>
    %cst_100 = arith.constant dense_resource<torch_tensor_768_torch.float32_25> : tensor<768xf32>
    %cst_101 = arith.constant dense_resource<torch_tensor_768_torch.float32_24> : tensor<768xf32>
    %cst_102 = arith.constant dense_resource<torch_tensor_768_torch.float32_23> : tensor<768xf32>
    %cst_103 = arith.constant dense_resource<torch_tensor_3072_768_torch.float32_3> : tensor<3072x768xf32>
    %cst_104 = arith.constant dense_resource<torch_tensor_3072_torch.float32_3> : tensor<3072xf32>
    %cst_105 = arith.constant dense_resource<torch_tensor_768_3072_torch.float32_3> : tensor<768x3072xf32>
    %cst_106 = arith.constant dense_resource<torch_tensor_768_torch.float32_22> : tensor<768xf32>
    %cst_107 = arith.constant dense_resource<torch_tensor_768_torch.float32_21> : tensor<768xf32>
    %cst_108 = arith.constant dense_resource<torch_tensor_768_torch.float32_20> : tensor<768xf32>
    %cst_109 = arith.constant dense_resource<torch_tensor_768_768_torch.float32_3> : tensor<768x768xf32>
    %cst_110 = arith.constant dense_resource<torch_tensor_2304_torch.float32_3> : tensor<2304xf32>
    %cst_111 = arith.constant dense_resource<torch_tensor_768_2304_torch.float32_3> : tensor<768x2304xf32>
    %cst_112 = arith.constant dense_resource<torch_tensor_768_torch.float32_19> : tensor<768xf32>
    %cst_113 = arith.constant dense_resource<torch_tensor_768_torch.float32_18> : tensor<768xf32>
    %cst_114 = arith.constant dense_resource<torch_tensor_768_torch.float32_17> : tensor<768xf32>
    %cst_115 = arith.constant dense_resource<torch_tensor_3072_768_torch.float32_2> : tensor<3072x768xf32>
    %cst_116 = arith.constant dense_resource<torch_tensor_3072_torch.float32_2> : tensor<3072xf32>
    %cst_117 = arith.constant dense_resource<torch_tensor_768_3072_torch.float32_2> : tensor<768x3072xf32>
    %cst_118 = arith.constant dense_resource<torch_tensor_768_torch.float32_16> : tensor<768xf32>
    %cst_119 = arith.constant dense_resource<torch_tensor_768_torch.float32_15> : tensor<768xf32>
    %cst_120 = arith.constant dense_resource<torch_tensor_768_torch.float32_14> : tensor<768xf32>
    %cst_121 = arith.constant dense_resource<torch_tensor_768_768_torch.float32_2> : tensor<768x768xf32>
    %cst_122 = arith.constant dense_resource<torch_tensor_2304_torch.float32_2> : tensor<2304xf32>
    %cst_123 = arith.constant dense_resource<torch_tensor_768_2304_torch.float32_2> : tensor<768x2304xf32>
    %cst_124 = arith.constant dense_resource<torch_tensor_768_torch.float32_13> : tensor<768xf32>
    %cst_125 = arith.constant dense_resource<torch_tensor_768_torch.float32_12> : tensor<768xf32>
    %cst_126 = arith.constant dense_resource<torch_tensor_768_torch.float32_11> : tensor<768xf32>
    %cst_127 = arith.constant dense_resource<torch_tensor_3072_768_torch.float32_1> : tensor<3072x768xf32>
    %cst_128 = arith.constant dense_resource<torch_tensor_3072_torch.float32_1> : tensor<3072xf32>
    %cst_129 = arith.constant dense_resource<torch_tensor_768_3072_torch.float32_1> : tensor<768x3072xf32>
    %cst_130 = arith.constant dense_resource<torch_tensor_768_torch.float32_10> : tensor<768xf32>
    %cst_131 = arith.constant dense_resource<torch_tensor_768_torch.float32_9> : tensor<768xf32>
    %cst_132 = arith.constant dense_resource<torch_tensor_768_torch.float32_8> : tensor<768xf32>
    %cst_133 = arith.constant dense_resource<torch_tensor_768_768_torch.float32_1> : tensor<768x768xf32>
    %cst_134 = arith.constant dense_resource<torch_tensor_2304_torch.float32_1> : tensor<2304xf32>
    %cst_135 = arith.constant dense_resource<torch_tensor_768_2304_torch.float32_1> : tensor<768x2304xf32>
    %cst_136 = arith.constant dense_resource<torch_tensor_768_torch.float32_7> : tensor<768xf32>
    %cst_137 = arith.constant dense_resource<torch_tensor_768_torch.float32_6> : tensor<768xf32>
    %cst_138 = arith.constant dense_resource<torch_tensor_768_torch.float32_5> : tensor<768xf32>
    %cst_139 = arith.constant dense_resource<torch_tensor_3072_768_torch.float32> : tensor<3072x768xf32>
    %cst_140 = arith.constant 0.79788456080286541 : f64
    %cst_141 = arith.constant 4.471500e-02 : f64
    %cst_142 = arith.constant dense_resource<torch_tensor_3072_torch.float32> : tensor<3072xf32>
    %cst_143 = arith.constant dense_resource<torch_tensor_768_3072_torch.float32> : tensor<768x3072xf32>
    %cst_144 = arith.constant dense_resource<torch_tensor_768_torch.float32_4> : tensor<768xf32>
    %cst_145 = arith.constant dense_resource<torch_tensor_768_torch.float32_3> : tensor<768xf32>
    %cst_146 = arith.constant dense_resource<torch_tensor_768_torch.float32_2> : tensor<768xf32>
    %cst_147 = arith.constant dense_resource<torch_tensor_768_768_torch.float32> : tensor<768x768xf32>
    %cst_148 = arith.constant 0.35355339059327379 : f64
    %cst_149 = arith.constant 0xFFF0000000000000 : f64
    %cst_150 = arith.constant dense_resource<torch_tensor_2304_torch.float32> : tensor<2304xf32>
    %cst_151 = arith.constant dense_resource<torch_tensor_768_2304_torch.float32> : tensor<768x2304xf32>
    %cst_152 = arith.constant dense_resource<torch_tensor_768_torch.float32_1> : tensor<768xf32>
    %cst_153 = arith.constant dense_resource<torch_tensor_768_torch.float32> : tensor<768xf32>
    %cst_154 = arith.constant 1.000000e-05 : f64
    %cst_155 = arith.constant dense_resource<torch_tensor_1024_768_torch.float32> : tensor<1024x768xf32>
    %cst_156 = arith.constant dense_resource<torch_tensor_50257_768_torch.float32> : tensor<50257x768xf32>
    %c2_i64 = arith.constant 2 : i64
    %true = arith.constant true
    %cst_157 = arith.constant 7.680000e+02 : f64
    %cst_158 = arith.constant 7.680000e+02 : f32
    %cst_159 = arith.constant 5.000000e-01 : f32
    %cst_160 = arith.constant 1.000000e+00 : f32
    %dim = tensor.dim %arg1, %c1 : tensor<1x?xi64>
    %0 = arith.index_cast %dim : index to i64
    %1 = arith.cmpi slt, %0, %c0_i64 : i64
    %2 = arith.select %1, %c1_i64, %0 : i64
    %3 = arith.select %1, %c2_i64, %c1_i64 : i64
    %4 = arith.cmpi sle, %3, %c1_i64 : i64
    cf.assert %4, "must have at most one inferred (negative) dimension"
    %dim_161 = tensor.dim %arg0, %c1 : tensor<1x?xi64>
    %5 = arith.index_cast %dim_161 : index to i64
    %6 = arith.divsi %5, %2 : i64
    %7 = arith.select %1, %6, %0 : i64
    %from_elements = tensor.from_elements %6, %7 : tensor<2xi64>
    %reshape = tensor.reshape %arg0(%from_elements) : (tensor<1x?xi64>, tensor<2xi64>) -> tensor<1x?xi64>
    %8 = arith.index_cast %7 : i64 to index
    %9 = tensor.empty(%8) : tensor<1x?x768xf32>
    %10 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%reshape : tensor<1x?xi64>) outs(%9 : tensor<1x?x768xf32>) {
    ^bb0(%in: i64, %out: f32):
      %1190 = arith.index_cast %in : i64 to index
      %1191 = linalg.index 2 : index
      %1192 = arith.cmpi slt, %1190, %c50257 : index
      cf.assert %1192, "index must be smaller than dim size"
      %1193 = arith.cmpi sge, %in, %c0_i64 : i64
      cf.assert %1193, "index must be larger or equal to 0"
      %extracted = tensor.extract %cst_156[%1190, %1191] : tensor<50257x768xf32>
      linalg.yield %extracted : f32
    } -> tensor<1x?x768xf32>
    %11 = tensor.empty(%dim) : tensor<?xi64>
    %12 = linalg.generic {indexing_maps = [#map2], iterator_types = ["parallel"]} outs(%11 : tensor<?xi64>) {
    ^bb0(%out: i64):
      %1190 = linalg.index 0 : index
      %1191 = arith.index_cast %1190 : index to i64
      linalg.yield %1191 : i64
    } -> tensor<?xi64>
    %expanded = tensor.expand_shape %12 [[0, 1]] output_shape [1, %dim] : tensor<?xi64> into tensor<1x?xi64>
    %13 = tensor.empty(%dim) : tensor<1x?x768xf32>
    %14 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%expanded : tensor<1x?xi64>) outs(%13 : tensor<1x?x768xf32>) {
    ^bb0(%in: i64, %out: f32):
      %1190 = arith.index_cast %in : i64 to index
      %1191 = linalg.index 2 : index
      %1192 = arith.cmpi slt, %1190, %c1024 : index
      cf.assert %1192, "index must be smaller than dim size"
      %1193 = arith.cmpi sge, %in, %c0_i64 : i64
      cf.assert %1193, "index must be larger or equal to 0"
      %extracted = tensor.extract %cst_155[%1190, %1191] : tensor<1024x768xf32>
      linalg.yield %extracted : f32
    } -> tensor<1x?x768xf32>
    %15 = arith.cmpi eq, %8, %dim : index
    cf.assert %15, "mismatched size for broadcast"
    %16 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%10, %14 : tensor<1x?x768xf32>, tensor<1x?x768xf32>) outs(%9 : tensor<1x?x768xf32>) {
    ^bb0(%in: f32, %in_495: f32, %out: f32):
      %1190 = arith.addf %in, %in_495 : f32
      linalg.yield %1190 : f32
    } -> tensor<1x?x768xf32>
    %17 = tensor.empty(%dim) : tensor<1x?xi1>
    %18 = linalg.generic {indexing_maps = [#map3, #map3], iterator_types = ["parallel", "parallel"]} ins(%arg1 : tensor<1x?xi64>) outs(%17 : tensor<1x?xi1>) {
    ^bb0(%in: i64, %out: i1):
      %1190 = arith.cmpi ne, %in, %c0_i64 : i64
      linalg.yield %1190 : i1
    } -> tensor<1x?xi1>
    %19 = tensor.empty() : tensor<1xi64>
    %20 = linalg.generic {indexing_maps = [#map2], iterator_types = ["parallel"]} outs(%19 : tensor<1xi64>) {
    ^bb0(%out: i64):
      linalg.yield %c0_i64 : i64
    } -> tensor<1xi64>
    %expanded_162 = tensor.expand_shape %20 [[0, 1, 2, 3]] output_shape [1, 1, 1, 1] : tensor<1xi64> into tensor<1x1x1x1xi64>
    %expanded_163 = tensor.expand_shape %12 [[0, 1, 2, 3]] output_shape [1, 1, %dim, 1] : tensor<?xi64> into tensor<1x1x?x1xi64>
    %expanded_164 = tensor.expand_shape %12 [[0, 1, 2, 3]] output_shape [1, 1, 1, %dim] : tensor<?xi64> into tensor<1x1x1x?xi64>
    %21 = tensor.empty() : tensor<i1>
    %22 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = []} ins(%21 : tensor<i1>) outs(%21 : tensor<i1>) {
    ^bb0(%in: i1, %out: i1):
      linalg.yield %true : i1
    } -> tensor<i1>
    %23 = tensor.empty(%dim, %dim) : tensor<1x1x?x?xi1>
    %24 = linalg.generic {indexing_maps = [#map5, #map6, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded_164, %expanded_163 : tensor<1x1x1x?xi64>, tensor<1x1x?x1xi64>) outs(%23 : tensor<1x1x?x?xi1>) {
    ^bb0(%in: i64, %in_495: i64, %out: i1):
      %1190 = arith.cmpi sle, %in, %in_495 : i64
      linalg.yield %1190 : i1
    } -> tensor<1x1x?x?xi1>
    %25 = linalg.generic {indexing_maps = [#map8, #map7, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%22, %24 : tensor<i1>, tensor<1x1x?x?xi1>) outs(%23 : tensor<1x1x?x?xi1>) {
    ^bb0(%in: i1, %in_495: i1, %out: i1):
      %1190 = arith.andi %in, %in_495 : i1
      linalg.yield %1190 : i1
    } -> tensor<1x1x?x?xi1>
    %dim_165 = tensor.dim %expanded_164, %c3 : tensor<1x1x1x?xi64>
    %26 = tensor.empty(%dim_165) : tensor<1x1x1x?xi1>
    %27 = linalg.generic {indexing_maps = [#map9, #map10, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded_162, %expanded_164 : tensor<1x1x1x1xi64>, tensor<1x1x1x?xi64>) outs(%26 : tensor<1x1x1x?xi1>) {
    ^bb0(%in: i64, %in_495: i64, %out: i1):
      %1190 = arith.cmpi slt, %in, %c0_i64 : i64
      %1191 = arith.addi %in, %c1_i64 : i64
      %1192 = arith.select %1190, %1191, %in : i64
      %1193 = arith.index_cast %1192 : i64 to index
      %1194 = arith.cmpi slt, %in_495, %c0_i64 : i64
      %1195 = arith.addi %in_495, %0 : i64
      %1196 = arith.select %1194, %1195, %in_495 : i64
      %1197 = arith.index_cast %1196 : i64 to index
      %extracted = tensor.extract %18[%1193, %1197] : tensor<1x?xi1>
      linalg.yield %extracted : i1
    } -> tensor<1x1x1x?xi1>
    %28 = arith.cmpi eq, %dim, %dim_165 : index
    cf.assert %28, "mismatched size for broadcast"
    %29 = linalg.generic {indexing_maps = [#map7, #map5, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%25, %27 : tensor<1x1x?x?xi1>, tensor<1x1x1x?xi1>) outs(%23 : tensor<1x1x?x?xi1>) {
    ^bb0(%in: i1, %in_495: i1, %out: i1):
      %1190 = arith.andi %in, %in_495 : i1
      linalg.yield %1190 : i1
    } -> tensor<1x1x?x?xi1>
    %30 = arith.cmpi sge, %0, %c0_i64 : i64
    cf.assert %30, "unimplemented: dynamic negative broadcast sizes"
    cf.assert %30, "unimplemented: dynamic negative broadcast sizes"
    %31 = tensor.empty(%dim, %dim) : tensor<1x1x?x?xi1>
    %32 = linalg.generic {indexing_maps = [#map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%31 : tensor<1x1x?x?xi1>) {
    ^bb0(%out: i1):
      %1190 = linalg.index 2 : index
      %1191 = linalg.index 3 : index
      %1192 = arith.cmpi eq, %dim, %c1 : index
      %1193 = arith.select %1192, %c0, %1190 : index
      %1194 = arith.select %1192, %c0, %1191 : index
      %extracted = tensor.extract %29[%c0, %c0, %1193, %1194] : tensor<1x1x?x?xi1>
      linalg.yield %extracted : i1
    } -> tensor<1x1x?x?xi1>
    %33 = tensor.empty(%8) : tensor<1x?x768xf64>
    %34 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%16 : tensor<1x?x768xf32>) outs(%33 : tensor<1x?x768xf64>) {
    ^bb0(%in: f32, %out: f64):
      %1190 = arith.extf %in : f32 to f64
      linalg.yield %1190 : f64
    } -> tensor<1x?x768xf64>
    %35 = tensor.empty(%8) : tensor<1x?x1xf64>
    %36 = linalg.fill ins(%cst : f64) outs(%35 : tensor<1x?x1xf64>) -> tensor<1x?x1xf64>
    %37 = linalg.generic {indexing_maps = [#map1, #map11], iterator_types = ["parallel", "parallel", "reduction"]} ins(%34 : tensor<1x?x768xf64>) outs(%36 : tensor<1x?x1xf64>) {
    ^bb0(%in: f64, %out: f64):
      %1190 = arith.addf %in, %out : f64
      linalg.yield %1190 : f64
    } -> tensor<1x?x1xf64>
    %38 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%37 : tensor<1x?x1xf64>) outs(%35 : tensor<1x?x1xf64>) {
    ^bb0(%in: f64, %out: f64):
      %1190 = arith.divf %in, %cst_157 : f64
      linalg.yield %1190 : f64
    } -> tensor<1x?x1xf64>
    %39 = linalg.generic {indexing_maps = [#map1, #map11, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%34, %38 : tensor<1x?x768xf64>, tensor<1x?x1xf64>) outs(%33 : tensor<1x?x768xf64>) {
    ^bb0(%in: f64, %in_495: f64, %out: f64):
      %1190 = arith.subf %in, %in_495 : f64
      linalg.yield %1190 : f64
    } -> tensor<1x?x768xf64>
    %40 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%39, %39 : tensor<1x?x768xf64>, tensor<1x?x768xf64>) outs(%33 : tensor<1x?x768xf64>) {
    ^bb0(%in: f64, %in_495: f64, %out: f64):
      %1190 = arith.mulf %in, %in_495 : f64
      linalg.yield %1190 : f64
    } -> tensor<1x?x768xf64>
    %41 = linalg.generic {indexing_maps = [#map1, #map11], iterator_types = ["parallel", "parallel", "reduction"]} ins(%40 : tensor<1x?x768xf64>) outs(%36 : tensor<1x?x1xf64>) {
    ^bb0(%in: f64, %out: f64):
      %1190 = arith.addf %in, %out : f64
      linalg.yield %1190 : f64
    } -> tensor<1x?x1xf64>
    %42 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%41 : tensor<1x?x1xf64>) outs(%35 : tensor<1x?x1xf64>) {
    ^bb0(%in: f64, %out: f64):
      %1190 = arith.divf %in, %cst_157 : f64
      linalg.yield %1190 : f64
    } -> tensor<1x?x1xf64>
    %43 = tensor.empty(%8) : tensor<1x?x1xf32>
    %44 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%42 : tensor<1x?x1xf64>) outs(%43 : tensor<1x?x1xf32>) {
    ^bb0(%in: f64, %out: f32):
      %1190 = arith.truncf %in : f64 to f32
      linalg.yield %1190 : f32
    } -> tensor<1x?x1xf32>
    %45 = linalg.fill ins(%cst_0 : f32) outs(%43 : tensor<1x?x1xf32>) -> tensor<1x?x1xf32>
    %46 = linalg.generic {indexing_maps = [#map1, #map11], iterator_types = ["parallel", "parallel", "reduction"]} ins(%16 : tensor<1x?x768xf32>) outs(%45 : tensor<1x?x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1190 = arith.addf %in, %out : f32
      linalg.yield %1190 : f32
    } -> tensor<1x?x1xf32>
    %47 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%46 : tensor<1x?x1xf32>) outs(%43 : tensor<1x?x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1190 = arith.divf %in, %cst_158 : f32
      linalg.yield %1190 : f32
    } -> tensor<1x?x1xf32>
    %48 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%44 : tensor<1x?x1xf32>) outs(%43 : tensor<1x?x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1190 = arith.truncf %cst_154 : f64 to f32
      %1191 = arith.addf %in, %1190 : f32
      linalg.yield %1191 : f32
    } -> tensor<1x?x1xf32>
    %49 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%48 : tensor<1x?x1xf32>) outs(%43 : tensor<1x?x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1190 = math.rsqrt %in : f32
      linalg.yield %1190 : f32
    } -> tensor<1x?x1xf32>
    %50 = linalg.generic {indexing_maps = [#map1, #map11, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%16, %47 : tensor<1x?x768xf32>, tensor<1x?x1xf32>) outs(%9 : tensor<1x?x768xf32>) {
    ^bb0(%in: f32, %in_495: f32, %out: f32):
      %1190 = arith.subf %in, %in_495 : f32
      linalg.yield %1190 : f32
    } -> tensor<1x?x768xf32>
    %51 = linalg.generic {indexing_maps = [#map1, #map11, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%50, %49 : tensor<1x?x768xf32>, tensor<1x?x1xf32>) outs(%9 : tensor<1x?x768xf32>) {
    ^bb0(%in: f32, %in_495: f32, %out: f32):
      %1190 = arith.mulf %in, %in_495 : f32
      linalg.yield %1190 : f32
    } -> tensor<1x?x768xf32>
    %52 = linalg.generic {indexing_maps = [#map1, #map12, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%51, %cst_153 : tensor<1x?x768xf32>, tensor<768xf32>) outs(%9 : tensor<1x?x768xf32>) {
    ^bb0(%in: f32, %in_495: f32, %out: f32):
      %1190 = arith.mulf %in, %in_495 : f32
      linalg.yield %1190 : f32
    } -> tensor<1x?x768xf32>
    %53 = linalg.generic {indexing_maps = [#map1, #map12, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%52, %cst_152 : tensor<1x?x768xf32>, tensor<768xf32>) outs(%9 : tensor<1x?x768xf32>) {
    ^bb0(%in: f32, %in_495: f32, %out: f32):
      %1190 = arith.addf %in, %in_495 : f32
      linalg.yield %1190 : f32
    } -> tensor<1x?x768xf32>
    %collapsed = tensor.collapse_shape %53 [[0, 1], [2]] : tensor<1x?x768xf32> into tensor<?x768xf32>
    %54 = arith.index_cast %7 : i64 to index
    %55 = tensor.empty(%54) : tensor<?x2304xf32>
    %56 = linalg.fill ins(%cst_0 : f32) outs(%55 : tensor<?x2304xf32>) -> tensor<?x2304xf32>
    %57 = linalg.matmul ins(%collapsed, %cst_151 : tensor<?x768xf32>, tensor<768x2304xf32>) outs(%56 : tensor<?x2304xf32>) -> tensor<?x2304xf32>
    %58 = linalg.generic {indexing_maps = [#map3, #map13, #map3], iterator_types = ["parallel", "parallel"]} ins(%57, %cst_150 : tensor<?x2304xf32>, tensor<2304xf32>) outs(%55 : tensor<?x2304xf32>) {
    ^bb0(%in: f32, %in_495: f32, %out: f32):
      %1190 = arith.addf %in, %in_495 : f32
      linalg.yield %1190 : f32
    } -> tensor<?x2304xf32>
    %expanded_166 = tensor.expand_shape %58 [[0, 1], [2]] output_shape [1, %54, 2304] : tensor<?x2304xf32> into tensor<1x?x2304xf32>
    %extracted_slice = tensor.extract_slice %expanded_166[0, 0, 0] [1, %54, 768] [1, 1, 1] : tensor<1x?x2304xf32> to tensor<1x?x768xf32>
    %extracted_slice_167 = tensor.extract_slice %expanded_166[0, 0, 768] [1, %54, 768] [1, 1, 1] : tensor<1x?x2304xf32> to tensor<1x?x768xf32>
    %extracted_slice_168 = tensor.extract_slice %expanded_166[0, 0, 1536] [1, %54, 768] [1, 1, 1] : tensor<1x?x2304xf32> to tensor<1x?x768xf32>
    %59 = arith.extui %1 : i1 to i64
    %60 = arith.addi %59, %c1_i64 : i64
    %61 = arith.muli %2, %c64_i64 : i64
    %62 = arith.cmpi sle, %60, %c1_i64 : i64
    cf.assert %62, "must have at most one inferred (negative) dimension"
    %63 = arith.muli %7, %c768_i64 : i64
    %64 = arith.divsi %63, %61 : i64
    %65 = arith.select %1, %64, %0 : i64
    %from_elements_169 = tensor.from_elements %c1_i64, %65, %64, %c64_i64 : tensor<4xi64>
    %reshape_170 = tensor.reshape %extracted_slice_167(%from_elements_169) : (tensor<1x?x768xf32>, tensor<4xi64>) -> tensor<1x?x12x64xf32>
    %66 = arith.index_cast %65 : i64 to index
    cf.assert %62, "must have at most one inferred (negative) dimension"
    %reshape_171 = tensor.reshape %extracted_slice_168(%from_elements_169) : (tensor<1x?x768xf32>, tensor<4xi64>) -> tensor<1x?x12x64xf32>
    %67 = tensor.empty(%66) : tensor<1x12x?x64xf32>
    %transposed = linalg.transpose ins(%reshape_171 : tensor<1x?x12x64xf32>) outs(%67 : tensor<1x12x?x64xf32>) permutation = [0, 2, 1, 3] 
    cf.assert %62, "must have at most one inferred (negative) dimension"
    %reshape_172 = tensor.reshape %extracted_slice(%from_elements_169) : (tensor<1x?x768xf32>, tensor<4xi64>) -> tensor<1x?x12x64xf32>
    %transposed_173 = linalg.transpose ins(%reshape_172 : tensor<1x?x12x64xf32>) outs(%67 : tensor<1x12x?x64xf32>) permutation = [0, 2, 1, 3] 
    %68 = tensor.empty(%dim, %dim) : tensor<1x1x?x?xf32>
    %69 = linalg.generic {indexing_maps = [#map7, #map8, #map8, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%32, %cst_2, %cst_3 : tensor<1x1x?x?xi1>, tensor<f32>, tensor<f32>) outs(%68 : tensor<1x1x?x?xf32>) {
    ^bb0(%in: i1, %in_495: f32, %in_496: f32, %out: f32):
      %1190 = arith.select %in, %in_495, %in_496 : f32
      linalg.yield %1190 : f32
    } -> tensor<1x1x?x?xf32>
    %70 = linalg.generic {indexing_maps = [#map7, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%transposed_173 : tensor<1x12x?x64xf32>) outs(%67 : tensor<1x12x?x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1190 = arith.truncf %cst_148 : f64 to f32
      %1191 = arith.mulf %in, %1190 : f32
      linalg.yield %1191 : f32
    } -> tensor<1x12x?x64xf32>
    %71 = tensor.empty(%66) : tensor<1x12x64x?xf32>
    %transposed_174 = linalg.transpose ins(%reshape_170 : tensor<1x?x12x64xf32>) outs(%71 : tensor<1x12x64x?xf32>) permutation = [0, 2, 3, 1] 
    %72 = linalg.generic {indexing_maps = [#map7, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%transposed_174 : tensor<1x12x64x?xf32>) outs(%71 : tensor<1x12x64x?xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1190 = arith.truncf %cst_148 : f64 to f32
      %1191 = arith.mulf %in, %1190 : f32
      linalg.yield %1191 : f32
    } -> tensor<1x12x64x?xf32>
    cf.assert %30, "unimplemented: dynamic negative broadcast sizes"
    %73 = tensor.empty(%dim) : tensor<1x12x?x64xf32>
    %74 = linalg.generic {indexing_maps = [#map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%73 : tensor<1x12x?x64xf32>) {
    ^bb0(%out: f32):
      %1190 = linalg.index 1 : index
      %1191 = linalg.index 2 : index
      %1192 = linalg.index 3 : index
      %1193 = arith.cmpi eq, %66, %c1 : index
      %1194 = arith.select %1193, %c0, %1191 : index
      %extracted = tensor.extract %70[%c0, %1190, %1194, %1192] : tensor<1x12x?x64xf32>
      linalg.yield %extracted : f32
    } -> tensor<1x12x?x64xf32>
    %collapsed_175 = tensor.collapse_shape %74 [[0, 1], [2], [3]] : tensor<1x12x?x64xf32> into tensor<12x?x64xf32>
    cf.assert %30, "unimplemented: dynamic negative broadcast sizes"
    %75 = tensor.empty(%dim) : tensor<1x12x64x?xf32>
    %76 = linalg.generic {indexing_maps = [#map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%75 : tensor<1x12x64x?xf32>) {
    ^bb0(%out: f32):
      %1190 = linalg.index 1 : index
      %1191 = linalg.index 2 : index
      %1192 = linalg.index 3 : index
      %1193 = arith.cmpi eq, %66, %c1 : index
      %1194 = arith.select %1193, %c0, %1192 : index
      %extracted = tensor.extract %72[%c0, %1190, %1191, %1194] : tensor<1x12x64x?xf32>
      linalg.yield %extracted : f32
    } -> tensor<1x12x64x?xf32>
    %collapsed_176 = tensor.collapse_shape %76 [[0, 1], [2], [3]] : tensor<1x12x64x?xf32> into tensor<12x64x?xf32>
    %77 = tensor.empty(%dim, %dim) : tensor<12x?x?xf32>
    %78 = linalg.fill ins(%cst_0 : f32) outs(%77 : tensor<12x?x?xf32>) -> tensor<12x?x?xf32>
    %79 = linalg.batch_matmul ins(%collapsed_175, %collapsed_176 : tensor<12x?x64xf32>, tensor<12x64x?xf32>) outs(%78 : tensor<12x?x?xf32>) -> tensor<12x?x?xf32>
    %80 = arith.muli %0, %c12_i64 : i64
    %81 = arith.select %1, %c12_i64, %80 : i64
    %82 = arith.muli %81, %0 : i64
    %83 = arith.select %1, %c12_i64, %82 : i64
    %84 = arith.select %1, %60, %59 : i64
    %85 = arith.cmpi sle, %84, %c1_i64 : i64
    cf.assert %85, "must have at most one inferred (negative) dimension"
    %86 = arith.muli %80, %0 : i64
    %87 = arith.divsi %86, %83 : i64
    %88 = arith.select %1, %87, %0 : i64
    %from_elements_177 = tensor.from_elements %c1_i64, %c12_i64, %88, %88 : tensor<4xi64>
    %reshape_178 = tensor.reshape %79(%from_elements_177) : (tensor<12x?x?xf32>, tensor<4xi64>) -> tensor<1x12x?x?xf32>
    %89 = arith.index_cast %88 : i64 to index
    %90 = arith.cmpi eq, %89, %dim : index
    cf.assert %90, "mismatched size for broadcast"
    cf.assert %90, "mismatched size for broadcast"
    %91 = tensor.empty(%89, %89) : tensor<1x12x?x?xf32>
    %92 = linalg.generic {indexing_maps = [#map7, #map14, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%reshape_178, %69 : tensor<1x12x?x?xf32>, tensor<1x1x?x?xf32>) outs(%91 : tensor<1x12x?x?xf32>) {
    ^bb0(%in: f32, %in_495: f32, %out: f32):
      %1190 = arith.addf %in, %in_495 : f32
      linalg.yield %1190 : f32
    } -> tensor<1x12x?x?xf32>
    %93 = tensor.empty(%89) : tensor<1x12x?xi64>
    %94 = linalg.fill ins(%c0_i64 : i64) outs(%93 : tensor<1x12x?xi64>) -> tensor<1x12x?xi64>
    %95 = tensor.empty(%89) : tensor<1x12x?xf32>
    %96 = linalg.fill ins(%cst_1 : f32) outs(%95 : tensor<1x12x?xf32>) -> tensor<1x12x?xf32>
    %97:2 = linalg.generic {indexing_maps = [#map7, #map15, #map15], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%92 : tensor<1x12x?x?xf32>) outs(%96, %94 : tensor<1x12x?xf32>, tensor<1x12x?xi64>) {
    ^bb0(%in: f32, %out: f32, %out_495: i64):
      %1190 = linalg.index 3 : index
      %1191 = arith.index_cast %1190 : index to i64
      %1192 = arith.maximumf %in, %out : f32
      %1193 = arith.cmpf ogt, %in, %out : f32
      %1194 = arith.select %1193, %1191, %out_495 : i64
      linalg.yield %1192, %1194 : f32, i64
    } -> (tensor<1x12x?xf32>, tensor<1x12x?xi64>)
    %expanded_179 = tensor.expand_shape %97#0 [[0], [1], [2, 3]] output_shape [1, 12, %89, 1] : tensor<1x12x?xf32> into tensor<1x12x?x1xf32>
    %98 = linalg.generic {indexing_maps = [#map7, #map6, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%92, %expanded_179 : tensor<1x12x?x?xf32>, tensor<1x12x?x1xf32>) outs(%91 : tensor<1x12x?x?xf32>) {
    ^bb0(%in: f32, %in_495: f32, %out: f32):
      %1190 = arith.subf %in, %in_495 : f32
      linalg.yield %1190 : f32
    } -> tensor<1x12x?x?xf32>
    %99 = linalg.generic {indexing_maps = [#map7, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%98 : tensor<1x12x?x?xf32>) outs(%91 : tensor<1x12x?x?xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1190 = math.exp %in : f32
      linalg.yield %1190 : f32
    } -> tensor<1x12x?x?xf32>
    %100 = tensor.empty(%89) : tensor<1x12x?x1xf32>
    %101 = linalg.fill ins(%cst_0 : f32) outs(%100 : tensor<1x12x?x1xf32>) -> tensor<1x12x?x1xf32>
    %102 = linalg.generic {indexing_maps = [#map7, #map6], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%99 : tensor<1x12x?x?xf32>) outs(%101 : tensor<1x12x?x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1190 = arith.addf %in, %out : f32
      linalg.yield %1190 : f32
    } -> tensor<1x12x?x1xf32>
    %103 = linalg.generic {indexing_maps = [#map7, #map6, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%99, %102 : tensor<1x12x?x?xf32>, tensor<1x12x?x1xf32>) outs(%91 : tensor<1x12x?x?xf32>) {
    ^bb0(%in: f32, %in_495: f32, %out: f32):
      %1190 = arith.divf %in, %in_495 : f32
      linalg.yield %1190 : f32
    } -> tensor<1x12x?x?xf32>
    %104 = tensor.empty(%89, %89) : tensor<1x12x?x?xi1>
    %105 = linalg.generic {indexing_maps = [#map7, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%92 : tensor<1x12x?x?xf32>) outs(%104 : tensor<1x12x?x?xi1>) {
    ^bb0(%in: f32, %out: i1):
      %1190 = arith.extf %in : f32 to f64
      %1191 = arith.cmpf oeq, %1190, %cst_149 : f64
      linalg.yield %1191 : i1
    } -> tensor<1x12x?x?xi1>
    %106 = linalg.generic {indexing_maps = [#map7, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%105 : tensor<1x12x?x?xi1>) outs(%104 : tensor<1x12x?x?xi1>) {
    ^bb0(%in: i1, %out: i1):
      %1190 = arith.cmpi eq, %in, %false : i1
      linalg.yield %1190 : i1
    } -> tensor<1x12x?x?xi1>
    %107 = tensor.empty(%89) : tensor<1x12x?x1xi1>
    %108 = linalg.fill ins(%false : i1) outs(%107 : tensor<1x12x?x1xi1>) -> tensor<1x12x?x1xi1>
    %109 = linalg.generic {indexing_maps = [#map7, #map6], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%106 : tensor<1x12x?x?xi1>) outs(%108 : tensor<1x12x?x1xi1>) {
    ^bb0(%in: i1, %out: i1):
      %1190 = arith.ori %in, %out : i1
      linalg.yield %1190 : i1
    } -> tensor<1x12x?x1xi1>
    %110 = linalg.generic {indexing_maps = [#map7, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%109 : tensor<1x12x?x1xi1>) outs(%107 : tensor<1x12x?x1xi1>) {
    ^bb0(%in: i1, %out: i1):
      %1190 = arith.cmpi eq, %in, %false : i1
      linalg.yield %1190 : i1
    } -> tensor<1x12x?x1xi1>
    %111 = arith.cmpi sge, %88, %c0_i64 : i64
    cf.assert %111, "negative values not allowed in new dimensions"
    %112 = arith.index_cast %88 : i64 to index
    cf.assert %111, "negative values not allowed in new dimensions"
    %113 = tensor.empty(%112, %112) : tensor<1x12x?x?xf32>
    %114 = linalg.generic {indexing_maps = [#map8, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%cst_2 : tensor<f32>) outs(%113 : tensor<1x12x?x?xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x12x?x?xf32>
    %115 = arith.cmpi eq, %89, %112 : index
    cf.assert %115, "mismatched size for broadcast"
    %116 = arith.cmpi eq, %112, %89 : index
    cf.assert %116, "mismatched size for broadcast"
    %117 = tensor.empty(%89, %112) : tensor<1x12x?x?xf32>
    %118 = linalg.generic {indexing_maps = [#map6, #map7, #map7, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%110, %114, %103 : tensor<1x12x?x1xi1>, tensor<1x12x?x?xf32>, tensor<1x12x?x?xf32>) outs(%117 : tensor<1x12x?x?xf32>) {
    ^bb0(%in: i1, %in_495: f32, %in_496: f32, %out: f32):
      %1190 = arith.select %in, %in_495, %in_496 : f32
      linalg.yield %1190 : f32
    } -> tensor<1x12x?x?xf32>
    cf.assert %30, "unimplemented: dynamic negative broadcast sizes"
    cf.assert %30, "unimplemented: dynamic negative broadcast sizes"
    %119 = tensor.empty(%dim, %dim) : tensor<1x12x?x?xf32>
    %120 = linalg.generic {indexing_maps = [#map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%119 : tensor<1x12x?x?xf32>) {
    ^bb0(%out: f32):
      %1190 = linalg.index 1 : index
      %1191 = linalg.index 2 : index
      %1192 = linalg.index 3 : index
      %1193 = arith.cmpi eq, %89, %c1 : index
      %1194 = arith.select %1193, %c0, %1191 : index
      %1195 = arith.cmpi eq, %112, %c1 : index
      %1196 = arith.select %1195, %c0, %1192 : index
      %extracted = tensor.extract %118[%c0, %1190, %1194, %1196] : tensor<1x12x?x?xf32>
      linalg.yield %extracted : f32
    } -> tensor<1x12x?x?xf32>
    cf.assert %85, "must have at most one inferred (negative) dimension"
    %from_elements_180 = tensor.from_elements %c12_i64, %88, %88 : tensor<3xi64>
    %reshape_181 = tensor.reshape %120(%from_elements_180) : (tensor<1x12x?x?xf32>, tensor<3xi64>) -> tensor<12x?x?xf32>
    cf.assert %30, "unimplemented: dynamic negative broadcast sizes"
    %121 = linalg.generic {indexing_maps = [#map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%73 : tensor<1x12x?x64xf32>) {
    ^bb0(%out: f32):
      %1190 = linalg.index 1 : index
      %1191 = linalg.index 2 : index
      %1192 = linalg.index 3 : index
      %1193 = arith.cmpi eq, %66, %c1 : index
      %1194 = arith.select %1193, %c0, %1191 : index
      %extracted = tensor.extract %transposed[%c0, %1190, %1194, %1192] : tensor<1x12x?x64xf32>
      linalg.yield %extracted : f32
    } -> tensor<1x12x?x64xf32>
    %collapsed_182 = tensor.collapse_shape %121 [[0, 1], [2], [3]] : tensor<1x12x?x64xf32> into tensor<12x?x64xf32>
    %122 = arith.cmpi eq, %88, %0 : i64
    cf.assert %122, "mismatching contracting dimension"
    %123 = tensor.empty(%89) : tensor<12x?x64xf32>
    %124 = linalg.fill ins(%cst_0 : f32) outs(%123 : tensor<12x?x64xf32>) -> tensor<12x?x64xf32>
    %125 = linalg.batch_matmul ins(%reshape_181, %collapsed_182 : tensor<12x?x?xf32>, tensor<12x?x64xf32>) outs(%124 : tensor<12x?x64xf32>) -> tensor<12x?x64xf32>
    %expanded_183 = tensor.expand_shape %125 [[0, 1], [2], [3]] output_shape [1, 12, %89, 64] : tensor<12x?x64xf32> into tensor<1x12x?x64xf32>
    %126 = tensor.empty(%89) : tensor<1x?x12x64xf32>
    %transposed_184 = linalg.transpose ins(%expanded_183 : tensor<1x12x?x64xf32>) outs(%126 : tensor<1x?x12x64xf32>) permutation = [0, 2, 1, 3] 
    cf.assert %62, "must have at most one inferred (negative) dimension"
    %127 = arith.muli %88, %c768_i64 : i64
    %128 = arith.divsi %127, %2 : i64
    %129 = arith.select %1, %128, %0 : i64
    %from_elements_185 = tensor.from_elements %c1_i64, %129, %128 : tensor<3xi64>
    %reshape_186 = tensor.reshape %transposed_184(%from_elements_185) : (tensor<1x?x12x64xf32>, tensor<3xi64>) -> tensor<1x?x768xf32>
    %collapsed_187 = tensor.collapse_shape %reshape_186 [[0, 1], [2]] : tensor<1x?x768xf32> into tensor<?x768xf32>
    %130 = arith.index_cast %129 : i64 to index
    %131 = tensor.empty(%130) : tensor<?x768xf32>
    %132 = linalg.fill ins(%cst_0 : f32) outs(%131 : tensor<?x768xf32>) -> tensor<?x768xf32>
    %133 = linalg.matmul ins(%collapsed_187, %cst_147 : tensor<?x768xf32>, tensor<768x768xf32>) outs(%132 : tensor<?x768xf32>) -> tensor<?x768xf32>
    %134 = linalg.generic {indexing_maps = [#map3, #map13, #map3], iterator_types = ["parallel", "parallel"]} ins(%133, %cst_146 : tensor<?x768xf32>, tensor<768xf32>) outs(%131 : tensor<?x768xf32>) {
    ^bb0(%in: f32, %in_495: f32, %out: f32):
      %1190 = arith.addf %in, %in_495 : f32
      linalg.yield %1190 : f32
    } -> tensor<?x768xf32>
    %expanded_188 = tensor.expand_shape %134 [[0, 1], [2]] output_shape [1, %130, 768] : tensor<?x768xf32> into tensor<1x?x768xf32>
    %135 = arith.cmpi eq, %130, %8 : index
    cf.assert %135, "mismatched size for broadcast"
    %136 = tensor.empty(%130) : tensor<1x?x768xf32>
    %137 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%expanded_188, %16 : tensor<1x?x768xf32>, tensor<1x?x768xf32>) outs(%136 : tensor<1x?x768xf32>) {
    ^bb0(%in: f32, %in_495: f32, %out: f32):
      %1190 = arith.addf %in, %in_495 : f32
      linalg.yield %1190 : f32
    } -> tensor<1x?x768xf32>
    %138 = tensor.empty(%130) : tensor<1x?x768xf64>
    %139 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%137 : tensor<1x?x768xf32>) outs(%138 : tensor<1x?x768xf64>) {
    ^bb0(%in: f32, %out: f64):
      %1190 = arith.extf %in : f32 to f64
      linalg.yield %1190 : f64
    } -> tensor<1x?x768xf64>
    %140 = tensor.empty(%130) : tensor<1x?x1xf64>
    %141 = linalg.fill ins(%cst : f64) outs(%140 : tensor<1x?x1xf64>) -> tensor<1x?x1xf64>
    %142 = linalg.generic {indexing_maps = [#map1, #map11], iterator_types = ["parallel", "parallel", "reduction"]} ins(%139 : tensor<1x?x768xf64>) outs(%141 : tensor<1x?x1xf64>) {
    ^bb0(%in: f64, %out: f64):
      %1190 = arith.addf %in, %out : f64
      linalg.yield %1190 : f64
    } -> tensor<1x?x1xf64>
    %143 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%142 : tensor<1x?x1xf64>) outs(%140 : tensor<1x?x1xf64>) {
    ^bb0(%in: f64, %out: f64):
      %1190 = arith.divf %in, %cst_157 : f64
      linalg.yield %1190 : f64
    } -> tensor<1x?x1xf64>
    %144 = linalg.generic {indexing_maps = [#map1, #map11, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%139, %143 : tensor<1x?x768xf64>, tensor<1x?x1xf64>) outs(%138 : tensor<1x?x768xf64>) {
    ^bb0(%in: f64, %in_495: f64, %out: f64):
      %1190 = arith.subf %in, %in_495 : f64
      linalg.yield %1190 : f64
    } -> tensor<1x?x768xf64>
    %145 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%144, %144 : tensor<1x?x768xf64>, tensor<1x?x768xf64>) outs(%138 : tensor<1x?x768xf64>) {
    ^bb0(%in: f64, %in_495: f64, %out: f64):
      %1190 = arith.mulf %in, %in_495 : f64
      linalg.yield %1190 : f64
    } -> tensor<1x?x768xf64>
    %146 = linalg.generic {indexing_maps = [#map1, #map11], iterator_types = ["parallel", "parallel", "reduction"]} ins(%145 : tensor<1x?x768xf64>) outs(%141 : tensor<1x?x1xf64>) {
    ^bb0(%in: f64, %out: f64):
      %1190 = arith.addf %in, %out : f64
      linalg.yield %1190 : f64
    } -> tensor<1x?x1xf64>
    %147 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%146 : tensor<1x?x1xf64>) outs(%140 : tensor<1x?x1xf64>) {
    ^bb0(%in: f64, %out: f64):
      %1190 = arith.divf %in, %cst_157 : f64
      linalg.yield %1190 : f64
    } -> tensor<1x?x1xf64>
    %148 = tensor.empty(%130) : tensor<1x?x1xf32>
    %149 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%147 : tensor<1x?x1xf64>) outs(%148 : tensor<1x?x1xf32>) {
    ^bb0(%in: f64, %out: f32):
      %1190 = arith.truncf %in : f64 to f32
      linalg.yield %1190 : f32
    } -> tensor<1x?x1xf32>
    %150 = linalg.fill ins(%cst_0 : f32) outs(%148 : tensor<1x?x1xf32>) -> tensor<1x?x1xf32>
    %151 = linalg.generic {indexing_maps = [#map1, #map11], iterator_types = ["parallel", "parallel", "reduction"]} ins(%137 : tensor<1x?x768xf32>) outs(%150 : tensor<1x?x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1190 = arith.addf %in, %out : f32
      linalg.yield %1190 : f32
    } -> tensor<1x?x1xf32>
    %152 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%151 : tensor<1x?x1xf32>) outs(%148 : tensor<1x?x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1190 = arith.divf %in, %cst_158 : f32
      linalg.yield %1190 : f32
    } -> tensor<1x?x1xf32>
    %153 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%149 : tensor<1x?x1xf32>) outs(%148 : tensor<1x?x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1190 = arith.truncf %cst_154 : f64 to f32
      %1191 = arith.addf %in, %1190 : f32
      linalg.yield %1191 : f32
    } -> tensor<1x?x1xf32>
    %154 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%153 : tensor<1x?x1xf32>) outs(%148 : tensor<1x?x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1190 = math.rsqrt %in : f32
      linalg.yield %1190 : f32
    } -> tensor<1x?x1xf32>
    %155 = linalg.generic {indexing_maps = [#map1, #map11, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%137, %152 : tensor<1x?x768xf32>, tensor<1x?x1xf32>) outs(%136 : tensor<1x?x768xf32>) {
    ^bb0(%in: f32, %in_495: f32, %out: f32):
      %1190 = arith.subf %in, %in_495 : f32
      linalg.yield %1190 : f32
    } -> tensor<1x?x768xf32>
    %156 = linalg.generic {indexing_maps = [#map1, #map11, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%155, %154 : tensor<1x?x768xf32>, tensor<1x?x1xf32>) outs(%136 : tensor<1x?x768xf32>) {
    ^bb0(%in: f32, %in_495: f32, %out: f32):
      %1190 = arith.mulf %in, %in_495 : f32
      linalg.yield %1190 : f32
    } -> tensor<1x?x768xf32>
    %157 = linalg.generic {indexing_maps = [#map1, #map12, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%156, %cst_145 : tensor<1x?x768xf32>, tensor<768xf32>) outs(%136 : tensor<1x?x768xf32>) {
    ^bb0(%in: f32, %in_495: f32, %out: f32):
      %1190 = arith.mulf %in, %in_495 : f32
      linalg.yield %1190 : f32
    } -> tensor<1x?x768xf32>
    %158 = linalg.generic {indexing_maps = [#map1, #map12, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%157, %cst_144 : tensor<1x?x768xf32>, tensor<768xf32>) outs(%136 : tensor<1x?x768xf32>) {
    ^bb0(%in: f32, %in_495: f32, %out: f32):
      %1190 = arith.addf %in, %in_495 : f32
      linalg.yield %1190 : f32
    } -> tensor<1x?x768xf32>
    %collapsed_189 = tensor.collapse_shape %158 [[0, 1], [2]] : tensor<1x?x768xf32> into tensor<?x768xf32>
    %159 = tensor.empty(%130) : tensor<?x3072xf32>
    %160 = linalg.fill ins(%cst_0 : f32) outs(%159 : tensor<?x3072xf32>) -> tensor<?x3072xf32>
    %161 = linalg.matmul ins(%collapsed_189, %cst_143 : tensor<?x768xf32>, tensor<768x3072xf32>) outs(%160 : tensor<?x3072xf32>) -> tensor<?x3072xf32>
    %162 = linalg.generic {indexing_maps = [#map3, #map13, #map3], iterator_types = ["parallel", "parallel"]} ins(%161, %cst_142 : tensor<?x3072xf32>, tensor<3072xf32>) outs(%159 : tensor<?x3072xf32>) {
    ^bb0(%in: f32, %in_495: f32, %out: f32):
      %1190 = arith.addf %in, %in_495 : f32
      linalg.yield %1190 : f32
    } -> tensor<?x3072xf32>
    %expanded_190 = tensor.expand_shape %162 [[0, 1], [2]] output_shape [1, %130, 3072] : tensor<?x3072xf32> into tensor<1x?x3072xf32>
    %163 = tensor.empty(%130) : tensor<1x?x3072xf32>
    %164 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%expanded_190 : tensor<1x?x3072xf32>) outs(%163 : tensor<1x?x3072xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1190 = arith.mulf %in, %cst_159 : f32
      linalg.yield %1190 : f32
    } -> tensor<1x?x3072xf32>
    %165 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%expanded_190 : tensor<1x?x3072xf32>) outs(%163 : tensor<1x?x3072xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1190 = math.fpowi %in, %c3_i64 : f32, i64
      linalg.yield %1190 : f32
    } -> tensor<1x?x3072xf32>
    %166 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%165 : tensor<1x?x3072xf32>) outs(%163 : tensor<1x?x3072xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1190 = arith.truncf %cst_141 : f64 to f32
      %1191 = arith.mulf %in, %1190 : f32
      linalg.yield %1191 : f32
    } -> tensor<1x?x3072xf32>
    %167 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%expanded_190, %166 : tensor<1x?x3072xf32>, tensor<1x?x3072xf32>) outs(%163 : tensor<1x?x3072xf32>) {
    ^bb0(%in: f32, %in_495: f32, %out: f32):
      %1190 = arith.addf %in, %in_495 : f32
      linalg.yield %1190 : f32
    } -> tensor<1x?x3072xf32>
    %168 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%167 : tensor<1x?x3072xf32>) outs(%163 : tensor<1x?x3072xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1190 = arith.truncf %cst_140 : f64 to f32
      %1191 = arith.mulf %in, %1190 : f32
      linalg.yield %1191 : f32
    } -> tensor<1x?x3072xf32>
    %169 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%168 : tensor<1x?x3072xf32>) outs(%163 : tensor<1x?x3072xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1190 = math.tanh %in : f32
      linalg.yield %1190 : f32
    } -> tensor<1x?x3072xf32>
    %170 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%169 : tensor<1x?x3072xf32>) outs(%163 : tensor<1x?x3072xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1190 = arith.addf %in, %cst_160 : f32
      linalg.yield %1190 : f32
    } -> tensor<1x?x3072xf32>
    %171 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%164, %170 : tensor<1x?x3072xf32>, tensor<1x?x3072xf32>) outs(%163 : tensor<1x?x3072xf32>) {
    ^bb0(%in: f32, %in_495: f32, %out: f32):
      %1190 = arith.mulf %in, %in_495 : f32
      linalg.yield %1190 : f32
    } -> tensor<1x?x3072xf32>
    %collapsed_191 = tensor.collapse_shape %171 [[0, 1], [2]] : tensor<1x?x3072xf32> into tensor<?x3072xf32>
    %172 = linalg.matmul ins(%collapsed_191, %cst_139 : tensor<?x3072xf32>, tensor<3072x768xf32>) outs(%132 : tensor<?x768xf32>) -> tensor<?x768xf32>
    %173 = linalg.generic {indexing_maps = [#map3, #map13, #map3], iterator_types = ["parallel", "parallel"]} ins(%172, %cst_138 : tensor<?x768xf32>, tensor<768xf32>) outs(%131 : tensor<?x768xf32>) {
    ^bb0(%in: f32, %in_495: f32, %out: f32):
      %1190 = arith.addf %in, %in_495 : f32
      linalg.yield %1190 : f32
    } -> tensor<?x768xf32>
    %expanded_192 = tensor.expand_shape %173 [[0, 1], [2]] output_shape [1, %130, 768] : tensor<?x768xf32> into tensor<1x?x768xf32>
    %174 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%137, %expanded_192 : tensor<1x?x768xf32>, tensor<1x?x768xf32>) outs(%136 : tensor<1x?x768xf32>) {
    ^bb0(%in: f32, %in_495: f32, %out: f32):
      %1190 = arith.addf %in, %in_495 : f32
      linalg.yield %1190 : f32
    } -> tensor<1x?x768xf32>
    %175 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%174 : tensor<1x?x768xf32>) outs(%138 : tensor<1x?x768xf64>) {
    ^bb0(%in: f32, %out: f64):
      %1190 = arith.extf %in : f32 to f64
      linalg.yield %1190 : f64
    } -> tensor<1x?x768xf64>
    %176 = linalg.generic {indexing_maps = [#map1, #map11], iterator_types = ["parallel", "parallel", "reduction"]} ins(%175 : tensor<1x?x768xf64>) outs(%141 : tensor<1x?x1xf64>) {
    ^bb0(%in: f64, %out: f64):
      %1190 = arith.addf %in, %out : f64
      linalg.yield %1190 : f64
    } -> tensor<1x?x1xf64>
    %177 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%176 : tensor<1x?x1xf64>) outs(%140 : tensor<1x?x1xf64>) {
    ^bb0(%in: f64, %out: f64):
      %1190 = arith.divf %in, %cst_157 : f64
      linalg.yield %1190 : f64
    } -> tensor<1x?x1xf64>
    %178 = linalg.generic {indexing_maps = [#map1, #map11, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%175, %177 : tensor<1x?x768xf64>, tensor<1x?x1xf64>) outs(%138 : tensor<1x?x768xf64>) {
    ^bb0(%in: f64, %in_495: f64, %out: f64):
      %1190 = arith.subf %in, %in_495 : f64
      linalg.yield %1190 : f64
    } -> tensor<1x?x768xf64>
    %179 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%178, %178 : tensor<1x?x768xf64>, tensor<1x?x768xf64>) outs(%138 : tensor<1x?x768xf64>) {
    ^bb0(%in: f64, %in_495: f64, %out: f64):
      %1190 = arith.mulf %in, %in_495 : f64
      linalg.yield %1190 : f64
    } -> tensor<1x?x768xf64>
    %180 = linalg.generic {indexing_maps = [#map1, #map11], iterator_types = ["parallel", "parallel", "reduction"]} ins(%179 : tensor<1x?x768xf64>) outs(%141 : tensor<1x?x1xf64>) {
    ^bb0(%in: f64, %out: f64):
      %1190 = arith.addf %in, %out : f64
      linalg.yield %1190 : f64
    } -> tensor<1x?x1xf64>
    %181 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%180 : tensor<1x?x1xf64>) outs(%140 : tensor<1x?x1xf64>) {
    ^bb0(%in: f64, %out: f64):
      %1190 = arith.divf %in, %cst_157 : f64
      linalg.yield %1190 : f64
    } -> tensor<1x?x1xf64>
    %182 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%181 : tensor<1x?x1xf64>) outs(%148 : tensor<1x?x1xf32>) {
    ^bb0(%in: f64, %out: f32):
      %1190 = arith.truncf %in : f64 to f32
      linalg.yield %1190 : f32
    } -> tensor<1x?x1xf32>
    %183 = linalg.generic {indexing_maps = [#map1, #map11], iterator_types = ["parallel", "parallel", "reduction"]} ins(%174 : tensor<1x?x768xf32>) outs(%150 : tensor<1x?x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1190 = arith.addf %in, %out : f32
      linalg.yield %1190 : f32
    } -> tensor<1x?x1xf32>
    %184 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%183 : tensor<1x?x1xf32>) outs(%148 : tensor<1x?x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1190 = arith.divf %in, %cst_158 : f32
      linalg.yield %1190 : f32
    } -> tensor<1x?x1xf32>
    %185 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%182 : tensor<1x?x1xf32>) outs(%148 : tensor<1x?x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1190 = arith.truncf %cst_154 : f64 to f32
      %1191 = arith.addf %in, %1190 : f32
      linalg.yield %1191 : f32
    } -> tensor<1x?x1xf32>
    %186 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%185 : tensor<1x?x1xf32>) outs(%148 : tensor<1x?x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1190 = math.rsqrt %in : f32
      linalg.yield %1190 : f32
    } -> tensor<1x?x1xf32>
    %187 = linalg.generic {indexing_maps = [#map1, #map11, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%174, %184 : tensor<1x?x768xf32>, tensor<1x?x1xf32>) outs(%136 : tensor<1x?x768xf32>) {
    ^bb0(%in: f32, %in_495: f32, %out: f32):
      %1190 = arith.subf %in, %in_495 : f32
      linalg.yield %1190 : f32
    } -> tensor<1x?x768xf32>
    %188 = linalg.generic {indexing_maps = [#map1, #map11, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%187, %186 : tensor<1x?x768xf32>, tensor<1x?x1xf32>) outs(%136 : tensor<1x?x768xf32>) {
    ^bb0(%in: f32, %in_495: f32, %out: f32):
      %1190 = arith.mulf %in, %in_495 : f32
      linalg.yield %1190 : f32
    } -> tensor<1x?x768xf32>
    %189 = linalg.generic {indexing_maps = [#map1, #map12, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%188, %cst_137 : tensor<1x?x768xf32>, tensor<768xf32>) outs(%136 : tensor<1x?x768xf32>) {
    ^bb0(%in: f32, %in_495: f32, %out: f32):
      %1190 = arith.mulf %in, %in_495 : f32
      linalg.yield %1190 : f32
    } -> tensor<1x?x768xf32>
    %190 = linalg.generic {indexing_maps = [#map1, #map12, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%189, %cst_136 : tensor<1x?x768xf32>, tensor<768xf32>) outs(%136 : tensor<1x?x768xf32>) {
    ^bb0(%in: f32, %in_495: f32, %out: f32):
      %1190 = arith.addf %in, %in_495 : f32
      linalg.yield %1190 : f32
    } -> tensor<1x?x768xf32>
    %collapsed_193 = tensor.collapse_shape %190 [[0, 1], [2]] : tensor<1x?x768xf32> into tensor<?x768xf32>
    %191 = tensor.empty(%130) : tensor<?x2304xf32>
    %192 = linalg.fill ins(%cst_0 : f32) outs(%191 : tensor<?x2304xf32>) -> tensor<?x2304xf32>
    %193 = linalg.matmul ins(%collapsed_193, %cst_135 : tensor<?x768xf32>, tensor<768x2304xf32>) outs(%192 : tensor<?x2304xf32>) -> tensor<?x2304xf32>
    %194 = linalg.generic {indexing_maps = [#map3, #map13, #map3], iterator_types = ["parallel", "parallel"]} ins(%193, %cst_134 : tensor<?x2304xf32>, tensor<2304xf32>) outs(%191 : tensor<?x2304xf32>) {
    ^bb0(%in: f32, %in_495: f32, %out: f32):
      %1190 = arith.addf %in, %in_495 : f32
      linalg.yield %1190 : f32
    } -> tensor<?x2304xf32>
    %expanded_194 = tensor.expand_shape %194 [[0, 1], [2]] output_shape [1, %130, 2304] : tensor<?x2304xf32> into tensor<1x?x2304xf32>
    %extracted_slice_195 = tensor.extract_slice %expanded_194[0, 0, 0] [1, %130, 768] [1, 1, 1] : tensor<1x?x2304xf32> to tensor<1x?x768xf32>
    %extracted_slice_196 = tensor.extract_slice %expanded_194[0, 0, 768] [1, %130, 768] [1, 1, 1] : tensor<1x?x2304xf32> to tensor<1x?x768xf32>
    %extracted_slice_197 = tensor.extract_slice %expanded_194[0, 0, 1536] [1, %130, 768] [1, 1, 1] : tensor<1x?x2304xf32> to tensor<1x?x768xf32>
    cf.assert %62, "must have at most one inferred (negative) dimension"
    %195 = arith.muli %129, %c768_i64 : i64
    %196 = arith.divsi %195, %61 : i64
    %197 = arith.select %1, %196, %0 : i64
    %from_elements_198 = tensor.from_elements %c1_i64, %197, %196, %c64_i64 : tensor<4xi64>
    %reshape_199 = tensor.reshape %extracted_slice_196(%from_elements_198) : (tensor<1x?x768xf32>, tensor<4xi64>) -> tensor<1x?x12x64xf32>
    %198 = arith.index_cast %197 : i64 to index
    cf.assert %62, "must have at most one inferred (negative) dimension"
    %reshape_200 = tensor.reshape %extracted_slice_197(%from_elements_198) : (tensor<1x?x768xf32>, tensor<4xi64>) -> tensor<1x?x12x64xf32>
    %199 = tensor.empty(%198) : tensor<1x12x?x64xf32>
    %transposed_201 = linalg.transpose ins(%reshape_200 : tensor<1x?x12x64xf32>) outs(%199 : tensor<1x12x?x64xf32>) permutation = [0, 2, 1, 3] 
    cf.assert %62, "must have at most one inferred (negative) dimension"
    %reshape_202 = tensor.reshape %extracted_slice_195(%from_elements_198) : (tensor<1x?x768xf32>, tensor<4xi64>) -> tensor<1x?x12x64xf32>
    %transposed_203 = linalg.transpose ins(%reshape_202 : tensor<1x?x12x64xf32>) outs(%199 : tensor<1x12x?x64xf32>) permutation = [0, 2, 1, 3] 
    %200 = linalg.generic {indexing_maps = [#map7, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%transposed_203 : tensor<1x12x?x64xf32>) outs(%199 : tensor<1x12x?x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1190 = arith.truncf %cst_148 : f64 to f32
      %1191 = arith.mulf %in, %1190 : f32
      linalg.yield %1191 : f32
    } -> tensor<1x12x?x64xf32>
    %201 = tensor.empty(%198) : tensor<1x12x64x?xf32>
    %transposed_204 = linalg.transpose ins(%reshape_199 : tensor<1x?x12x64xf32>) outs(%201 : tensor<1x12x64x?xf32>) permutation = [0, 2, 3, 1] 
    %202 = linalg.generic {indexing_maps = [#map7, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%transposed_204 : tensor<1x12x64x?xf32>) outs(%201 : tensor<1x12x64x?xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1190 = arith.truncf %cst_148 : f64 to f32
      %1191 = arith.mulf %in, %1190 : f32
      linalg.yield %1191 : f32
    } -> tensor<1x12x64x?xf32>
    cf.assert %30, "unimplemented: dynamic negative broadcast sizes"
    %203 = linalg.generic {indexing_maps = [#map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%73 : tensor<1x12x?x64xf32>) {
    ^bb0(%out: f32):
      %1190 = linalg.index 1 : index
      %1191 = linalg.index 2 : index
      %1192 = linalg.index 3 : index
      %1193 = arith.cmpi eq, %198, %c1 : index
      %1194 = arith.select %1193, %c0, %1191 : index
      %extracted = tensor.extract %200[%c0, %1190, %1194, %1192] : tensor<1x12x?x64xf32>
      linalg.yield %extracted : f32
    } -> tensor<1x12x?x64xf32>
    %collapsed_205 = tensor.collapse_shape %203 [[0, 1], [2], [3]] : tensor<1x12x?x64xf32> into tensor<12x?x64xf32>
    cf.assert %30, "unimplemented: dynamic negative broadcast sizes"
    %204 = linalg.generic {indexing_maps = [#map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%75 : tensor<1x12x64x?xf32>) {
    ^bb0(%out: f32):
      %1190 = linalg.index 1 : index
      %1191 = linalg.index 2 : index
      %1192 = linalg.index 3 : index
      %1193 = arith.cmpi eq, %198, %c1 : index
      %1194 = arith.select %1193, %c0, %1192 : index
      %extracted = tensor.extract %202[%c0, %1190, %1191, %1194] : tensor<1x12x64x?xf32>
      linalg.yield %extracted : f32
    } -> tensor<1x12x64x?xf32>
    %collapsed_206 = tensor.collapse_shape %204 [[0, 1], [2], [3]] : tensor<1x12x64x?xf32> into tensor<12x64x?xf32>
    %205 = linalg.batch_matmul ins(%collapsed_205, %collapsed_206 : tensor<12x?x64xf32>, tensor<12x64x?xf32>) outs(%78 : tensor<12x?x?xf32>) -> tensor<12x?x?xf32>
    cf.assert %85, "must have at most one inferred (negative) dimension"
    %reshape_207 = tensor.reshape %205(%from_elements_177) : (tensor<12x?x?xf32>, tensor<4xi64>) -> tensor<1x12x?x?xf32>
    cf.assert %90, "mismatched size for broadcast"
    cf.assert %90, "mismatched size for broadcast"
    %206 = linalg.generic {indexing_maps = [#map7, #map14, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%reshape_207, %69 : tensor<1x12x?x?xf32>, tensor<1x1x?x?xf32>) outs(%91 : tensor<1x12x?x?xf32>) {
    ^bb0(%in: f32, %in_495: f32, %out: f32):
      %1190 = arith.addf %in, %in_495 : f32
      linalg.yield %1190 : f32
    } -> tensor<1x12x?x?xf32>
    %207:2 = linalg.generic {indexing_maps = [#map7, #map15, #map15], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%206 : tensor<1x12x?x?xf32>) outs(%96, %94 : tensor<1x12x?xf32>, tensor<1x12x?xi64>) {
    ^bb0(%in: f32, %out: f32, %out_495: i64):
      %1190 = linalg.index 3 : index
      %1191 = arith.index_cast %1190 : index to i64
      %1192 = arith.maximumf %in, %out : f32
      %1193 = arith.cmpf ogt, %in, %out : f32
      %1194 = arith.select %1193, %1191, %out_495 : i64
      linalg.yield %1192, %1194 : f32, i64
    } -> (tensor<1x12x?xf32>, tensor<1x12x?xi64>)
    %expanded_208 = tensor.expand_shape %207#0 [[0], [1], [2, 3]] output_shape [1, 12, %89, 1] : tensor<1x12x?xf32> into tensor<1x12x?x1xf32>
    %208 = linalg.generic {indexing_maps = [#map7, #map6, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%206, %expanded_208 : tensor<1x12x?x?xf32>, tensor<1x12x?x1xf32>) outs(%91 : tensor<1x12x?x?xf32>) {
    ^bb0(%in: f32, %in_495: f32, %out: f32):
      %1190 = arith.subf %in, %in_495 : f32
      linalg.yield %1190 : f32
    } -> tensor<1x12x?x?xf32>
    %209 = linalg.generic {indexing_maps = [#map7, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%208 : tensor<1x12x?x?xf32>) outs(%91 : tensor<1x12x?x?xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1190 = math.exp %in : f32
      linalg.yield %1190 : f32
    } -> tensor<1x12x?x?xf32>
    %210 = linalg.generic {indexing_maps = [#map7, #map6], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%209 : tensor<1x12x?x?xf32>) outs(%101 : tensor<1x12x?x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1190 = arith.addf %in, %out : f32
      linalg.yield %1190 : f32
    } -> tensor<1x12x?x1xf32>
    %211 = linalg.generic {indexing_maps = [#map7, #map6, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%209, %210 : tensor<1x12x?x?xf32>, tensor<1x12x?x1xf32>) outs(%91 : tensor<1x12x?x?xf32>) {
    ^bb0(%in: f32, %in_495: f32, %out: f32):
      %1190 = arith.divf %in, %in_495 : f32
      linalg.yield %1190 : f32
    } -> tensor<1x12x?x?xf32>
    %212 = linalg.generic {indexing_maps = [#map7, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%206 : tensor<1x12x?x?xf32>) outs(%104 : tensor<1x12x?x?xi1>) {
    ^bb0(%in: f32, %out: i1):
      %1190 = arith.extf %in : f32 to f64
      %1191 = arith.cmpf oeq, %1190, %cst_149 : f64
      linalg.yield %1191 : i1
    } -> tensor<1x12x?x?xi1>
    %213 = linalg.generic {indexing_maps = [#map7, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%212 : tensor<1x12x?x?xi1>) outs(%104 : tensor<1x12x?x?xi1>) {
    ^bb0(%in: i1, %out: i1):
      %1190 = arith.cmpi eq, %in, %false : i1
      linalg.yield %1190 : i1
    } -> tensor<1x12x?x?xi1>
    %214 = linalg.generic {indexing_maps = [#map7, #map6], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%213 : tensor<1x12x?x?xi1>) outs(%108 : tensor<1x12x?x1xi1>) {
    ^bb0(%in: i1, %out: i1):
      %1190 = arith.ori %in, %out : i1
      linalg.yield %1190 : i1
    } -> tensor<1x12x?x1xi1>
    %215 = linalg.generic {indexing_maps = [#map7, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%214 : tensor<1x12x?x1xi1>) outs(%107 : tensor<1x12x?x1xi1>) {
    ^bb0(%in: i1, %out: i1):
      %1190 = arith.cmpi eq, %in, %false : i1
      linalg.yield %1190 : i1
    } -> tensor<1x12x?x1xi1>
    cf.assert %111, "negative values not allowed in new dimensions"
    cf.assert %111, "negative values not allowed in new dimensions"
    cf.assert %115, "mismatched size for broadcast"
    cf.assert %116, "mismatched size for broadcast"
    %216 = linalg.generic {indexing_maps = [#map6, #map7, #map7, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%215, %114, %211 : tensor<1x12x?x1xi1>, tensor<1x12x?x?xf32>, tensor<1x12x?x?xf32>) outs(%117 : tensor<1x12x?x?xf32>) {
    ^bb0(%in: i1, %in_495: f32, %in_496: f32, %out: f32):
      %1190 = arith.select %in, %in_495, %in_496 : f32
      linalg.yield %1190 : f32
    } -> tensor<1x12x?x?xf32>
    cf.assert %30, "unimplemented: dynamic negative broadcast sizes"
    cf.assert %30, "unimplemented: dynamic negative broadcast sizes"
    %217 = linalg.generic {indexing_maps = [#map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%119 : tensor<1x12x?x?xf32>) {
    ^bb0(%out: f32):
      %1190 = linalg.index 1 : index
      %1191 = linalg.index 2 : index
      %1192 = linalg.index 3 : index
      %1193 = arith.cmpi eq, %89, %c1 : index
      %1194 = arith.select %1193, %c0, %1191 : index
      %1195 = arith.cmpi eq, %112, %c1 : index
      %1196 = arith.select %1195, %c0, %1192 : index
      %extracted = tensor.extract %216[%c0, %1190, %1194, %1196] : tensor<1x12x?x?xf32>
      linalg.yield %extracted : f32
    } -> tensor<1x12x?x?xf32>
    cf.assert %85, "must have at most one inferred (negative) dimension"
    %reshape_209 = tensor.reshape %217(%from_elements_180) : (tensor<1x12x?x?xf32>, tensor<3xi64>) -> tensor<12x?x?xf32>
    cf.assert %30, "unimplemented: dynamic negative broadcast sizes"
    %218 = linalg.generic {indexing_maps = [#map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%73 : tensor<1x12x?x64xf32>) {
    ^bb0(%out: f32):
      %1190 = linalg.index 1 : index
      %1191 = linalg.index 2 : index
      %1192 = linalg.index 3 : index
      %1193 = arith.cmpi eq, %198, %c1 : index
      %1194 = arith.select %1193, %c0, %1191 : index
      %extracted = tensor.extract %transposed_201[%c0, %1190, %1194, %1192] : tensor<1x12x?x64xf32>
      linalg.yield %extracted : f32
    } -> tensor<1x12x?x64xf32>
    %collapsed_210 = tensor.collapse_shape %218 [[0, 1], [2], [3]] : tensor<1x12x?x64xf32> into tensor<12x?x64xf32>
    cf.assert %122, "mismatching contracting dimension"
    %219 = linalg.batch_matmul ins(%reshape_209, %collapsed_210 : tensor<12x?x?xf32>, tensor<12x?x64xf32>) outs(%124 : tensor<12x?x64xf32>) -> tensor<12x?x64xf32>
    %expanded_211 = tensor.expand_shape %219 [[0, 1], [2], [3]] output_shape [1, 12, %89, 64] : tensor<12x?x64xf32> into tensor<1x12x?x64xf32>
    %transposed_212 = linalg.transpose ins(%expanded_211 : tensor<1x12x?x64xf32>) outs(%126 : tensor<1x?x12x64xf32>) permutation = [0, 2, 1, 3] 
    cf.assert %62, "must have at most one inferred (negative) dimension"
    %reshape_213 = tensor.reshape %transposed_212(%from_elements_185) : (tensor<1x?x12x64xf32>, tensor<3xi64>) -> tensor<1x?x768xf32>
    %collapsed_214 = tensor.collapse_shape %reshape_213 [[0, 1], [2]] : tensor<1x?x768xf32> into tensor<?x768xf32>
    %220 = arith.index_cast %129 : i64 to index
    %221 = tensor.empty(%220) : tensor<?x768xf32>
    %222 = linalg.fill ins(%cst_0 : f32) outs(%221 : tensor<?x768xf32>) -> tensor<?x768xf32>
    %223 = linalg.matmul ins(%collapsed_214, %cst_133 : tensor<?x768xf32>, tensor<768x768xf32>) outs(%222 : tensor<?x768xf32>) -> tensor<?x768xf32>
    %224 = linalg.generic {indexing_maps = [#map3, #map13, #map3], iterator_types = ["parallel", "parallel"]} ins(%223, %cst_132 : tensor<?x768xf32>, tensor<768xf32>) outs(%221 : tensor<?x768xf32>) {
    ^bb0(%in: f32, %in_495: f32, %out: f32):
      %1190 = arith.addf %in, %in_495 : f32
      linalg.yield %1190 : f32
    } -> tensor<?x768xf32>
    %expanded_215 = tensor.expand_shape %224 [[0, 1], [2]] output_shape [1, %220, 768] : tensor<?x768xf32> into tensor<1x?x768xf32>
    %225 = arith.cmpi eq, %220, %130 : index
    cf.assert %225, "mismatched size for broadcast"
    %226 = tensor.empty(%220) : tensor<1x?x768xf32>
    %227 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%expanded_215, %174 : tensor<1x?x768xf32>, tensor<1x?x768xf32>) outs(%226 : tensor<1x?x768xf32>) {
    ^bb0(%in: f32, %in_495: f32, %out: f32):
      %1190 = arith.addf %in, %in_495 : f32
      linalg.yield %1190 : f32
    } -> tensor<1x?x768xf32>
    %228 = tensor.empty(%220) : tensor<1x?x768xf64>
    %229 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%227 : tensor<1x?x768xf32>) outs(%228 : tensor<1x?x768xf64>) {
    ^bb0(%in: f32, %out: f64):
      %1190 = arith.extf %in : f32 to f64
      linalg.yield %1190 : f64
    } -> tensor<1x?x768xf64>
    %230 = tensor.empty(%220) : tensor<1x?x1xf64>
    %231 = linalg.fill ins(%cst : f64) outs(%230 : tensor<1x?x1xf64>) -> tensor<1x?x1xf64>
    %232 = linalg.generic {indexing_maps = [#map1, #map11], iterator_types = ["parallel", "parallel", "reduction"]} ins(%229 : tensor<1x?x768xf64>) outs(%231 : tensor<1x?x1xf64>) {
    ^bb0(%in: f64, %out: f64):
      %1190 = arith.addf %in, %out : f64
      linalg.yield %1190 : f64
    } -> tensor<1x?x1xf64>
    %233 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%232 : tensor<1x?x1xf64>) outs(%230 : tensor<1x?x1xf64>) {
    ^bb0(%in: f64, %out: f64):
      %1190 = arith.divf %in, %cst_157 : f64
      linalg.yield %1190 : f64
    } -> tensor<1x?x1xf64>
    %234 = linalg.generic {indexing_maps = [#map1, #map11, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%229, %233 : tensor<1x?x768xf64>, tensor<1x?x1xf64>) outs(%228 : tensor<1x?x768xf64>) {
    ^bb0(%in: f64, %in_495: f64, %out: f64):
      %1190 = arith.subf %in, %in_495 : f64
      linalg.yield %1190 : f64
    } -> tensor<1x?x768xf64>
    %235 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%234, %234 : tensor<1x?x768xf64>, tensor<1x?x768xf64>) outs(%228 : tensor<1x?x768xf64>) {
    ^bb0(%in: f64, %in_495: f64, %out: f64):
      %1190 = arith.mulf %in, %in_495 : f64
      linalg.yield %1190 : f64
    } -> tensor<1x?x768xf64>
    %236 = linalg.generic {indexing_maps = [#map1, #map11], iterator_types = ["parallel", "parallel", "reduction"]} ins(%235 : tensor<1x?x768xf64>) outs(%231 : tensor<1x?x1xf64>) {
    ^bb0(%in: f64, %out: f64):
      %1190 = arith.addf %in, %out : f64
      linalg.yield %1190 : f64
    } -> tensor<1x?x1xf64>
    %237 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%236 : tensor<1x?x1xf64>) outs(%230 : tensor<1x?x1xf64>) {
    ^bb0(%in: f64, %out: f64):
      %1190 = arith.divf %in, %cst_157 : f64
      linalg.yield %1190 : f64
    } -> tensor<1x?x1xf64>
    %238 = tensor.empty(%220) : tensor<1x?x1xf32>
    %239 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%237 : tensor<1x?x1xf64>) outs(%238 : tensor<1x?x1xf32>) {
    ^bb0(%in: f64, %out: f32):
      %1190 = arith.truncf %in : f64 to f32
      linalg.yield %1190 : f32
    } -> tensor<1x?x1xf32>
    %240 = linalg.fill ins(%cst_0 : f32) outs(%238 : tensor<1x?x1xf32>) -> tensor<1x?x1xf32>
    %241 = linalg.generic {indexing_maps = [#map1, #map11], iterator_types = ["parallel", "parallel", "reduction"]} ins(%227 : tensor<1x?x768xf32>) outs(%240 : tensor<1x?x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1190 = arith.addf %in, %out : f32
      linalg.yield %1190 : f32
    } -> tensor<1x?x1xf32>
    %242 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%241 : tensor<1x?x1xf32>) outs(%238 : tensor<1x?x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1190 = arith.divf %in, %cst_158 : f32
      linalg.yield %1190 : f32
    } -> tensor<1x?x1xf32>
    %243 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%239 : tensor<1x?x1xf32>) outs(%238 : tensor<1x?x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1190 = arith.truncf %cst_154 : f64 to f32
      %1191 = arith.addf %in, %1190 : f32
      linalg.yield %1191 : f32
    } -> tensor<1x?x1xf32>
    %244 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%243 : tensor<1x?x1xf32>) outs(%238 : tensor<1x?x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1190 = math.rsqrt %in : f32
      linalg.yield %1190 : f32
    } -> tensor<1x?x1xf32>
    %245 = linalg.generic {indexing_maps = [#map1, #map11, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%227, %242 : tensor<1x?x768xf32>, tensor<1x?x1xf32>) outs(%226 : tensor<1x?x768xf32>) {
    ^bb0(%in: f32, %in_495: f32, %out: f32):
      %1190 = arith.subf %in, %in_495 : f32
      linalg.yield %1190 : f32
    } -> tensor<1x?x768xf32>
    %246 = linalg.generic {indexing_maps = [#map1, #map11, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%245, %244 : tensor<1x?x768xf32>, tensor<1x?x1xf32>) outs(%226 : tensor<1x?x768xf32>) {
    ^bb0(%in: f32, %in_495: f32, %out: f32):
      %1190 = arith.mulf %in, %in_495 : f32
      linalg.yield %1190 : f32
    } -> tensor<1x?x768xf32>
    %247 = linalg.generic {indexing_maps = [#map1, #map12, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%246, %cst_131 : tensor<1x?x768xf32>, tensor<768xf32>) outs(%226 : tensor<1x?x768xf32>) {
    ^bb0(%in: f32, %in_495: f32, %out: f32):
      %1190 = arith.mulf %in, %in_495 : f32
      linalg.yield %1190 : f32
    } -> tensor<1x?x768xf32>
    %248 = linalg.generic {indexing_maps = [#map1, #map12, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%247, %cst_130 : tensor<1x?x768xf32>, tensor<768xf32>) outs(%226 : tensor<1x?x768xf32>) {
    ^bb0(%in: f32, %in_495: f32, %out: f32):
      %1190 = arith.addf %in, %in_495 : f32
      linalg.yield %1190 : f32
    } -> tensor<1x?x768xf32>
    %collapsed_216 = tensor.collapse_shape %248 [[0, 1], [2]] : tensor<1x?x768xf32> into tensor<?x768xf32>
    %249 = tensor.empty(%220) : tensor<?x3072xf32>
    %250 = linalg.fill ins(%cst_0 : f32) outs(%249 : tensor<?x3072xf32>) -> tensor<?x3072xf32>
    %251 = linalg.matmul ins(%collapsed_216, %cst_129 : tensor<?x768xf32>, tensor<768x3072xf32>) outs(%250 : tensor<?x3072xf32>) -> tensor<?x3072xf32>
    %252 = linalg.generic {indexing_maps = [#map3, #map13, #map3], iterator_types = ["parallel", "parallel"]} ins(%251, %cst_128 : tensor<?x3072xf32>, tensor<3072xf32>) outs(%249 : tensor<?x3072xf32>) {
    ^bb0(%in: f32, %in_495: f32, %out: f32):
      %1190 = arith.addf %in, %in_495 : f32
      linalg.yield %1190 : f32
    } -> tensor<?x3072xf32>
    %expanded_217 = tensor.expand_shape %252 [[0, 1], [2]] output_shape [1, %220, 3072] : tensor<?x3072xf32> into tensor<1x?x3072xf32>
    %253 = tensor.empty(%220) : tensor<1x?x3072xf32>
    %254 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%expanded_217 : tensor<1x?x3072xf32>) outs(%253 : tensor<1x?x3072xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1190 = arith.mulf %in, %cst_159 : f32
      linalg.yield %1190 : f32
    } -> tensor<1x?x3072xf32>
    %255 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%expanded_217 : tensor<1x?x3072xf32>) outs(%253 : tensor<1x?x3072xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1190 = math.fpowi %in, %c3_i64 : f32, i64
      linalg.yield %1190 : f32
    } -> tensor<1x?x3072xf32>
    %256 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%255 : tensor<1x?x3072xf32>) outs(%253 : tensor<1x?x3072xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1190 = arith.truncf %cst_141 : f64 to f32
      %1191 = arith.mulf %in, %1190 : f32
      linalg.yield %1191 : f32
    } -> tensor<1x?x3072xf32>
    %257 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%expanded_217, %256 : tensor<1x?x3072xf32>, tensor<1x?x3072xf32>) outs(%253 : tensor<1x?x3072xf32>) {
    ^bb0(%in: f32, %in_495: f32, %out: f32):
      %1190 = arith.addf %in, %in_495 : f32
      linalg.yield %1190 : f32
    } -> tensor<1x?x3072xf32>
    %258 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%257 : tensor<1x?x3072xf32>) outs(%253 : tensor<1x?x3072xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1190 = arith.truncf %cst_140 : f64 to f32
      %1191 = arith.mulf %in, %1190 : f32
      linalg.yield %1191 : f32
    } -> tensor<1x?x3072xf32>
    %259 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%258 : tensor<1x?x3072xf32>) outs(%253 : tensor<1x?x3072xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1190 = math.tanh %in : f32
      linalg.yield %1190 : f32
    } -> tensor<1x?x3072xf32>
    %260 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%259 : tensor<1x?x3072xf32>) outs(%253 : tensor<1x?x3072xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1190 = arith.addf %in, %cst_160 : f32
      linalg.yield %1190 : f32
    } -> tensor<1x?x3072xf32>
    %261 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%254, %260 : tensor<1x?x3072xf32>, tensor<1x?x3072xf32>) outs(%253 : tensor<1x?x3072xf32>) {
    ^bb0(%in: f32, %in_495: f32, %out: f32):
      %1190 = arith.mulf %in, %in_495 : f32
      linalg.yield %1190 : f32
    } -> tensor<1x?x3072xf32>
    %collapsed_218 = tensor.collapse_shape %261 [[0, 1], [2]] : tensor<1x?x3072xf32> into tensor<?x3072xf32>
    %262 = linalg.matmul ins(%collapsed_218, %cst_127 : tensor<?x3072xf32>, tensor<3072x768xf32>) outs(%222 : tensor<?x768xf32>) -> tensor<?x768xf32>
    %263 = linalg.generic {indexing_maps = [#map3, #map13, #map3], iterator_types = ["parallel", "parallel"]} ins(%262, %cst_126 : tensor<?x768xf32>, tensor<768xf32>) outs(%221 : tensor<?x768xf32>) {
    ^bb0(%in: f32, %in_495: f32, %out: f32):
      %1190 = arith.addf %in, %in_495 : f32
      linalg.yield %1190 : f32
    } -> tensor<?x768xf32>
    %expanded_219 = tensor.expand_shape %263 [[0, 1], [2]] output_shape [1, %220, 768] : tensor<?x768xf32> into tensor<1x?x768xf32>
    %264 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%227, %expanded_219 : tensor<1x?x768xf32>, tensor<1x?x768xf32>) outs(%226 : tensor<1x?x768xf32>) {
    ^bb0(%in: f32, %in_495: f32, %out: f32):
      %1190 = arith.addf %in, %in_495 : f32
      linalg.yield %1190 : f32
    } -> tensor<1x?x768xf32>
    %265 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%264 : tensor<1x?x768xf32>) outs(%228 : tensor<1x?x768xf64>) {
    ^bb0(%in: f32, %out: f64):
      %1190 = arith.extf %in : f32 to f64
      linalg.yield %1190 : f64
    } -> tensor<1x?x768xf64>
    %266 = linalg.generic {indexing_maps = [#map1, #map11], iterator_types = ["parallel", "parallel", "reduction"]} ins(%265 : tensor<1x?x768xf64>) outs(%231 : tensor<1x?x1xf64>) {
    ^bb0(%in: f64, %out: f64):
      %1190 = arith.addf %in, %out : f64
      linalg.yield %1190 : f64
    } -> tensor<1x?x1xf64>
    %267 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%266 : tensor<1x?x1xf64>) outs(%230 : tensor<1x?x1xf64>) {
    ^bb0(%in: f64, %out: f64):
      %1190 = arith.divf %in, %cst_157 : f64
      linalg.yield %1190 : f64
    } -> tensor<1x?x1xf64>
    %268 = linalg.generic {indexing_maps = [#map1, #map11, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%265, %267 : tensor<1x?x768xf64>, tensor<1x?x1xf64>) outs(%228 : tensor<1x?x768xf64>) {
    ^bb0(%in: f64, %in_495: f64, %out: f64):
      %1190 = arith.subf %in, %in_495 : f64
      linalg.yield %1190 : f64
    } -> tensor<1x?x768xf64>
    %269 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%268, %268 : tensor<1x?x768xf64>, tensor<1x?x768xf64>) outs(%228 : tensor<1x?x768xf64>) {
    ^bb0(%in: f64, %in_495: f64, %out: f64):
      %1190 = arith.mulf %in, %in_495 : f64
      linalg.yield %1190 : f64
    } -> tensor<1x?x768xf64>
    %270 = linalg.generic {indexing_maps = [#map1, #map11], iterator_types = ["parallel", "parallel", "reduction"]} ins(%269 : tensor<1x?x768xf64>) outs(%231 : tensor<1x?x1xf64>) {
    ^bb0(%in: f64, %out: f64):
      %1190 = arith.addf %in, %out : f64
      linalg.yield %1190 : f64
    } -> tensor<1x?x1xf64>
    %271 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%270 : tensor<1x?x1xf64>) outs(%230 : tensor<1x?x1xf64>) {
    ^bb0(%in: f64, %out: f64):
      %1190 = arith.divf %in, %cst_157 : f64
      linalg.yield %1190 : f64
    } -> tensor<1x?x1xf64>
    %272 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%271 : tensor<1x?x1xf64>) outs(%238 : tensor<1x?x1xf32>) {
    ^bb0(%in: f64, %out: f32):
      %1190 = arith.truncf %in : f64 to f32
      linalg.yield %1190 : f32
    } -> tensor<1x?x1xf32>
    %273 = linalg.generic {indexing_maps = [#map1, #map11], iterator_types = ["parallel", "parallel", "reduction"]} ins(%264 : tensor<1x?x768xf32>) outs(%240 : tensor<1x?x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1190 = arith.addf %in, %out : f32
      linalg.yield %1190 : f32
    } -> tensor<1x?x1xf32>
    %274 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%273 : tensor<1x?x1xf32>) outs(%238 : tensor<1x?x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1190 = arith.divf %in, %cst_158 : f32
      linalg.yield %1190 : f32
    } -> tensor<1x?x1xf32>
    %275 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%272 : tensor<1x?x1xf32>) outs(%238 : tensor<1x?x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1190 = arith.truncf %cst_154 : f64 to f32
      %1191 = arith.addf %in, %1190 : f32
      linalg.yield %1191 : f32
    } -> tensor<1x?x1xf32>
    %276 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%275 : tensor<1x?x1xf32>) outs(%238 : tensor<1x?x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1190 = math.rsqrt %in : f32
      linalg.yield %1190 : f32
    } -> tensor<1x?x1xf32>
    %277 = linalg.generic {indexing_maps = [#map1, #map11, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%264, %274 : tensor<1x?x768xf32>, tensor<1x?x1xf32>) outs(%226 : tensor<1x?x768xf32>) {
    ^bb0(%in: f32, %in_495: f32, %out: f32):
      %1190 = arith.subf %in, %in_495 : f32
      linalg.yield %1190 : f32
    } -> tensor<1x?x768xf32>
    %278 = linalg.generic {indexing_maps = [#map1, #map11, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%277, %276 : tensor<1x?x768xf32>, tensor<1x?x1xf32>) outs(%226 : tensor<1x?x768xf32>) {
    ^bb0(%in: f32, %in_495: f32, %out: f32):
      %1190 = arith.mulf %in, %in_495 : f32
      linalg.yield %1190 : f32
    } -> tensor<1x?x768xf32>
    %279 = linalg.generic {indexing_maps = [#map1, #map12, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%278, %cst_125 : tensor<1x?x768xf32>, tensor<768xf32>) outs(%226 : tensor<1x?x768xf32>) {
    ^bb0(%in: f32, %in_495: f32, %out: f32):
      %1190 = arith.mulf %in, %in_495 : f32
      linalg.yield %1190 : f32
    } -> tensor<1x?x768xf32>
    %280 = linalg.generic {indexing_maps = [#map1, #map12, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%279, %cst_124 : tensor<1x?x768xf32>, tensor<768xf32>) outs(%226 : tensor<1x?x768xf32>) {
    ^bb0(%in: f32, %in_495: f32, %out: f32):
      %1190 = arith.addf %in, %in_495 : f32
      linalg.yield %1190 : f32
    } -> tensor<1x?x768xf32>
    %collapsed_220 = tensor.collapse_shape %280 [[0, 1], [2]] : tensor<1x?x768xf32> into tensor<?x768xf32>
    %281 = tensor.empty(%220) : tensor<?x2304xf32>
    %282 = linalg.fill ins(%cst_0 : f32) outs(%281 : tensor<?x2304xf32>) -> tensor<?x2304xf32>
    %283 = linalg.matmul ins(%collapsed_220, %cst_123 : tensor<?x768xf32>, tensor<768x2304xf32>) outs(%282 : tensor<?x2304xf32>) -> tensor<?x2304xf32>
    %284 = linalg.generic {indexing_maps = [#map3, #map13, #map3], iterator_types = ["parallel", "parallel"]} ins(%283, %cst_122 : tensor<?x2304xf32>, tensor<2304xf32>) outs(%281 : tensor<?x2304xf32>) {
    ^bb0(%in: f32, %in_495: f32, %out: f32):
      %1190 = arith.addf %in, %in_495 : f32
      linalg.yield %1190 : f32
    } -> tensor<?x2304xf32>
    %expanded_221 = tensor.expand_shape %284 [[0, 1], [2]] output_shape [1, %220, 2304] : tensor<?x2304xf32> into tensor<1x?x2304xf32>
    %extracted_slice_222 = tensor.extract_slice %expanded_221[0, 0, 0] [1, %220, 768] [1, 1, 1] : tensor<1x?x2304xf32> to tensor<1x?x768xf32>
    %extracted_slice_223 = tensor.extract_slice %expanded_221[0, 0, 768] [1, %220, 768] [1, 1, 1] : tensor<1x?x2304xf32> to tensor<1x?x768xf32>
    %extracted_slice_224 = tensor.extract_slice %expanded_221[0, 0, 1536] [1, %220, 768] [1, 1, 1] : tensor<1x?x2304xf32> to tensor<1x?x768xf32>
    cf.assert %62, "must have at most one inferred (negative) dimension"
    %285 = arith.muli %129, %c768_i64 : i64
    %286 = arith.divsi %285, %61 : i64
    %287 = arith.select %1, %286, %0 : i64
    %from_elements_225 = tensor.from_elements %c1_i64, %287, %286, %c64_i64 : tensor<4xi64>
    %reshape_226 = tensor.reshape %extracted_slice_223(%from_elements_225) : (tensor<1x?x768xf32>, tensor<4xi64>) -> tensor<1x?x12x64xf32>
    %288 = arith.index_cast %287 : i64 to index
    cf.assert %62, "must have at most one inferred (negative) dimension"
    %reshape_227 = tensor.reshape %extracted_slice_224(%from_elements_225) : (tensor<1x?x768xf32>, tensor<4xi64>) -> tensor<1x?x12x64xf32>
    %289 = tensor.empty(%288) : tensor<1x12x?x64xf32>
    %transposed_228 = linalg.transpose ins(%reshape_227 : tensor<1x?x12x64xf32>) outs(%289 : tensor<1x12x?x64xf32>) permutation = [0, 2, 1, 3] 
    cf.assert %62, "must have at most one inferred (negative) dimension"
    %reshape_229 = tensor.reshape %extracted_slice_222(%from_elements_225) : (tensor<1x?x768xf32>, tensor<4xi64>) -> tensor<1x?x12x64xf32>
    %transposed_230 = linalg.transpose ins(%reshape_229 : tensor<1x?x12x64xf32>) outs(%289 : tensor<1x12x?x64xf32>) permutation = [0, 2, 1, 3] 
    %290 = linalg.generic {indexing_maps = [#map7, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%transposed_230 : tensor<1x12x?x64xf32>) outs(%289 : tensor<1x12x?x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1190 = arith.truncf %cst_148 : f64 to f32
      %1191 = arith.mulf %in, %1190 : f32
      linalg.yield %1191 : f32
    } -> tensor<1x12x?x64xf32>
    %291 = tensor.empty(%288) : tensor<1x12x64x?xf32>
    %transposed_231 = linalg.transpose ins(%reshape_226 : tensor<1x?x12x64xf32>) outs(%291 : tensor<1x12x64x?xf32>) permutation = [0, 2, 3, 1] 
    %292 = linalg.generic {indexing_maps = [#map7, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%transposed_231 : tensor<1x12x64x?xf32>) outs(%291 : tensor<1x12x64x?xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1190 = arith.truncf %cst_148 : f64 to f32
      %1191 = arith.mulf %in, %1190 : f32
      linalg.yield %1191 : f32
    } -> tensor<1x12x64x?xf32>
    cf.assert %30, "unimplemented: dynamic negative broadcast sizes"
    %293 = linalg.generic {indexing_maps = [#map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%73 : tensor<1x12x?x64xf32>) {
    ^bb0(%out: f32):
      %1190 = linalg.index 1 : index
      %1191 = linalg.index 2 : index
      %1192 = linalg.index 3 : index
      %1193 = arith.cmpi eq, %288, %c1 : index
      %1194 = arith.select %1193, %c0, %1191 : index
      %extracted = tensor.extract %290[%c0, %1190, %1194, %1192] : tensor<1x12x?x64xf32>
      linalg.yield %extracted : f32
    } -> tensor<1x12x?x64xf32>
    %collapsed_232 = tensor.collapse_shape %293 [[0, 1], [2], [3]] : tensor<1x12x?x64xf32> into tensor<12x?x64xf32>
    cf.assert %30, "unimplemented: dynamic negative broadcast sizes"
    %294 = linalg.generic {indexing_maps = [#map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%75 : tensor<1x12x64x?xf32>) {
    ^bb0(%out: f32):
      %1190 = linalg.index 1 : index
      %1191 = linalg.index 2 : index
      %1192 = linalg.index 3 : index
      %1193 = arith.cmpi eq, %288, %c1 : index
      %1194 = arith.select %1193, %c0, %1192 : index
      %extracted = tensor.extract %292[%c0, %1190, %1191, %1194] : tensor<1x12x64x?xf32>
      linalg.yield %extracted : f32
    } -> tensor<1x12x64x?xf32>
    %collapsed_233 = tensor.collapse_shape %294 [[0, 1], [2], [3]] : tensor<1x12x64x?xf32> into tensor<12x64x?xf32>
    %295 = linalg.batch_matmul ins(%collapsed_232, %collapsed_233 : tensor<12x?x64xf32>, tensor<12x64x?xf32>) outs(%78 : tensor<12x?x?xf32>) -> tensor<12x?x?xf32>
    cf.assert %85, "must have at most one inferred (negative) dimension"
    %reshape_234 = tensor.reshape %295(%from_elements_177) : (tensor<12x?x?xf32>, tensor<4xi64>) -> tensor<1x12x?x?xf32>
    cf.assert %90, "mismatched size for broadcast"
    cf.assert %90, "mismatched size for broadcast"
    %296 = linalg.generic {indexing_maps = [#map7, #map14, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%reshape_234, %69 : tensor<1x12x?x?xf32>, tensor<1x1x?x?xf32>) outs(%91 : tensor<1x12x?x?xf32>) {
    ^bb0(%in: f32, %in_495: f32, %out: f32):
      %1190 = arith.addf %in, %in_495 : f32
      linalg.yield %1190 : f32
    } -> tensor<1x12x?x?xf32>
    %297:2 = linalg.generic {indexing_maps = [#map7, #map15, #map15], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%296 : tensor<1x12x?x?xf32>) outs(%96, %94 : tensor<1x12x?xf32>, tensor<1x12x?xi64>) {
    ^bb0(%in: f32, %out: f32, %out_495: i64):
      %1190 = linalg.index 3 : index
      %1191 = arith.index_cast %1190 : index to i64
      %1192 = arith.maximumf %in, %out : f32
      %1193 = arith.cmpf ogt, %in, %out : f32
      %1194 = arith.select %1193, %1191, %out_495 : i64
      linalg.yield %1192, %1194 : f32, i64
    } -> (tensor<1x12x?xf32>, tensor<1x12x?xi64>)
    %expanded_235 = tensor.expand_shape %297#0 [[0], [1], [2, 3]] output_shape [1, 12, %89, 1] : tensor<1x12x?xf32> into tensor<1x12x?x1xf32>
    %298 = linalg.generic {indexing_maps = [#map7, #map6, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%296, %expanded_235 : tensor<1x12x?x?xf32>, tensor<1x12x?x1xf32>) outs(%91 : tensor<1x12x?x?xf32>) {
    ^bb0(%in: f32, %in_495: f32, %out: f32):
      %1190 = arith.subf %in, %in_495 : f32
      linalg.yield %1190 : f32
    } -> tensor<1x12x?x?xf32>
    %299 = linalg.generic {indexing_maps = [#map7, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%298 : tensor<1x12x?x?xf32>) outs(%91 : tensor<1x12x?x?xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1190 = math.exp %in : f32
      linalg.yield %1190 : f32
    } -> tensor<1x12x?x?xf32>
    %300 = linalg.generic {indexing_maps = [#map7, #map6], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%299 : tensor<1x12x?x?xf32>) outs(%101 : tensor<1x12x?x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1190 = arith.addf %in, %out : f32
      linalg.yield %1190 : f32
    } -> tensor<1x12x?x1xf32>
    %301 = linalg.generic {indexing_maps = [#map7, #map6, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%299, %300 : tensor<1x12x?x?xf32>, tensor<1x12x?x1xf32>) outs(%91 : tensor<1x12x?x?xf32>) {
    ^bb0(%in: f32, %in_495: f32, %out: f32):
      %1190 = arith.divf %in, %in_495 : f32
      linalg.yield %1190 : f32
    } -> tensor<1x12x?x?xf32>
    %302 = linalg.generic {indexing_maps = [#map7, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%296 : tensor<1x12x?x?xf32>) outs(%104 : tensor<1x12x?x?xi1>) {
    ^bb0(%in: f32, %out: i1):
      %1190 = arith.extf %in : f32 to f64
      %1191 = arith.cmpf oeq, %1190, %cst_149 : f64
      linalg.yield %1191 : i1
    } -> tensor<1x12x?x?xi1>
    %303 = linalg.generic {indexing_maps = [#map7, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%302 : tensor<1x12x?x?xi1>) outs(%104 : tensor<1x12x?x?xi1>) {
    ^bb0(%in: i1, %out: i1):
      %1190 = arith.cmpi eq, %in, %false : i1
      linalg.yield %1190 : i1
    } -> tensor<1x12x?x?xi1>
    %304 = linalg.generic {indexing_maps = [#map7, #map6], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%303 : tensor<1x12x?x?xi1>) outs(%108 : tensor<1x12x?x1xi1>) {
    ^bb0(%in: i1, %out: i1):
      %1190 = arith.ori %in, %out : i1
      linalg.yield %1190 : i1
    } -> tensor<1x12x?x1xi1>
    %305 = linalg.generic {indexing_maps = [#map7, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%304 : tensor<1x12x?x1xi1>) outs(%107 : tensor<1x12x?x1xi1>) {
    ^bb0(%in: i1, %out: i1):
      %1190 = arith.cmpi eq, %in, %false : i1
      linalg.yield %1190 : i1
    } -> tensor<1x12x?x1xi1>
    cf.assert %111, "negative values not allowed in new dimensions"
    cf.assert %111, "negative values not allowed in new dimensions"
    cf.assert %115, "mismatched size for broadcast"
    cf.assert %116, "mismatched size for broadcast"
    %306 = linalg.generic {indexing_maps = [#map6, #map7, #map7, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%305, %114, %301 : tensor<1x12x?x1xi1>, tensor<1x12x?x?xf32>, tensor<1x12x?x?xf32>) outs(%117 : tensor<1x12x?x?xf32>) {
    ^bb0(%in: i1, %in_495: f32, %in_496: f32, %out: f32):
      %1190 = arith.select %in, %in_495, %in_496 : f32
      linalg.yield %1190 : f32
    } -> tensor<1x12x?x?xf32>
    cf.assert %30, "unimplemented: dynamic negative broadcast sizes"
    cf.assert %30, "unimplemented: dynamic negative broadcast sizes"
    %307 = linalg.generic {indexing_maps = [#map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%119 : tensor<1x12x?x?xf32>) {
    ^bb0(%out: f32):
      %1190 = linalg.index 1 : index
      %1191 = linalg.index 2 : index
      %1192 = linalg.index 3 : index
      %1193 = arith.cmpi eq, %89, %c1 : index
      %1194 = arith.select %1193, %c0, %1191 : index
      %1195 = arith.cmpi eq, %112, %c1 : index
      %1196 = arith.select %1195, %c0, %1192 : index
      %extracted = tensor.extract %306[%c0, %1190, %1194, %1196] : tensor<1x12x?x?xf32>
      linalg.yield %extracted : f32
    } -> tensor<1x12x?x?xf32>
    cf.assert %85, "must have at most one inferred (negative) dimension"
    %reshape_236 = tensor.reshape %307(%from_elements_180) : (tensor<1x12x?x?xf32>, tensor<3xi64>) -> tensor<12x?x?xf32>
    cf.assert %30, "unimplemented: dynamic negative broadcast sizes"
    %308 = linalg.generic {indexing_maps = [#map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%73 : tensor<1x12x?x64xf32>) {
    ^bb0(%out: f32):
      %1190 = linalg.index 1 : index
      %1191 = linalg.index 2 : index
      %1192 = linalg.index 3 : index
      %1193 = arith.cmpi eq, %288, %c1 : index
      %1194 = arith.select %1193, %c0, %1191 : index
      %extracted = tensor.extract %transposed_228[%c0, %1190, %1194, %1192] : tensor<1x12x?x64xf32>
      linalg.yield %extracted : f32
    } -> tensor<1x12x?x64xf32>
    %collapsed_237 = tensor.collapse_shape %308 [[0, 1], [2], [3]] : tensor<1x12x?x64xf32> into tensor<12x?x64xf32>
    cf.assert %122, "mismatching contracting dimension"
    %309 = linalg.batch_matmul ins(%reshape_236, %collapsed_237 : tensor<12x?x?xf32>, tensor<12x?x64xf32>) outs(%124 : tensor<12x?x64xf32>) -> tensor<12x?x64xf32>
    %expanded_238 = tensor.expand_shape %309 [[0, 1], [2], [3]] output_shape [1, 12, %89, 64] : tensor<12x?x64xf32> into tensor<1x12x?x64xf32>
    %transposed_239 = linalg.transpose ins(%expanded_238 : tensor<1x12x?x64xf32>) outs(%126 : tensor<1x?x12x64xf32>) permutation = [0, 2, 1, 3] 
    cf.assert %62, "must have at most one inferred (negative) dimension"
    %reshape_240 = tensor.reshape %transposed_239(%from_elements_185) : (tensor<1x?x12x64xf32>, tensor<3xi64>) -> tensor<1x?x768xf32>
    %collapsed_241 = tensor.collapse_shape %reshape_240 [[0, 1], [2]] : tensor<1x?x768xf32> into tensor<?x768xf32>
    %310 = arith.index_cast %129 : i64 to index
    %311 = tensor.empty(%310) : tensor<?x768xf32>
    %312 = linalg.fill ins(%cst_0 : f32) outs(%311 : tensor<?x768xf32>) -> tensor<?x768xf32>
    %313 = linalg.matmul ins(%collapsed_241, %cst_121 : tensor<?x768xf32>, tensor<768x768xf32>) outs(%312 : tensor<?x768xf32>) -> tensor<?x768xf32>
    %314 = linalg.generic {indexing_maps = [#map3, #map13, #map3], iterator_types = ["parallel", "parallel"]} ins(%313, %cst_120 : tensor<?x768xf32>, tensor<768xf32>) outs(%311 : tensor<?x768xf32>) {
    ^bb0(%in: f32, %in_495: f32, %out: f32):
      %1190 = arith.addf %in, %in_495 : f32
      linalg.yield %1190 : f32
    } -> tensor<?x768xf32>
    %expanded_242 = tensor.expand_shape %314 [[0, 1], [2]] output_shape [1, %310, 768] : tensor<?x768xf32> into tensor<1x?x768xf32>
    %315 = arith.cmpi eq, %310, %220 : index
    cf.assert %315, "mismatched size for broadcast"
    %316 = tensor.empty(%310) : tensor<1x?x768xf32>
    %317 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%expanded_242, %264 : tensor<1x?x768xf32>, tensor<1x?x768xf32>) outs(%316 : tensor<1x?x768xf32>) {
    ^bb0(%in: f32, %in_495: f32, %out: f32):
      %1190 = arith.addf %in, %in_495 : f32
      linalg.yield %1190 : f32
    } -> tensor<1x?x768xf32>
    %318 = tensor.empty(%310) : tensor<1x?x768xf64>
    %319 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%317 : tensor<1x?x768xf32>) outs(%318 : tensor<1x?x768xf64>) {
    ^bb0(%in: f32, %out: f64):
      %1190 = arith.extf %in : f32 to f64
      linalg.yield %1190 : f64
    } -> tensor<1x?x768xf64>
    %320 = tensor.empty(%310) : tensor<1x?x1xf64>
    %321 = linalg.fill ins(%cst : f64) outs(%320 : tensor<1x?x1xf64>) -> tensor<1x?x1xf64>
    %322 = linalg.generic {indexing_maps = [#map1, #map11], iterator_types = ["parallel", "parallel", "reduction"]} ins(%319 : tensor<1x?x768xf64>) outs(%321 : tensor<1x?x1xf64>) {
    ^bb0(%in: f64, %out: f64):
      %1190 = arith.addf %in, %out : f64
      linalg.yield %1190 : f64
    } -> tensor<1x?x1xf64>
    %323 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%322 : tensor<1x?x1xf64>) outs(%320 : tensor<1x?x1xf64>) {
    ^bb0(%in: f64, %out: f64):
      %1190 = arith.divf %in, %cst_157 : f64
      linalg.yield %1190 : f64
    } -> tensor<1x?x1xf64>
    %324 = linalg.generic {indexing_maps = [#map1, #map11, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%319, %323 : tensor<1x?x768xf64>, tensor<1x?x1xf64>) outs(%318 : tensor<1x?x768xf64>) {
    ^bb0(%in: f64, %in_495: f64, %out: f64):
      %1190 = arith.subf %in, %in_495 : f64
      linalg.yield %1190 : f64
    } -> tensor<1x?x768xf64>
    %325 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%324, %324 : tensor<1x?x768xf64>, tensor<1x?x768xf64>) outs(%318 : tensor<1x?x768xf64>) {
    ^bb0(%in: f64, %in_495: f64, %out: f64):
      %1190 = arith.mulf %in, %in_495 : f64
      linalg.yield %1190 : f64
    } -> tensor<1x?x768xf64>
    %326 = linalg.generic {indexing_maps = [#map1, #map11], iterator_types = ["parallel", "parallel", "reduction"]} ins(%325 : tensor<1x?x768xf64>) outs(%321 : tensor<1x?x1xf64>) {
    ^bb0(%in: f64, %out: f64):
      %1190 = arith.addf %in, %out : f64
      linalg.yield %1190 : f64
    } -> tensor<1x?x1xf64>
    %327 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%326 : tensor<1x?x1xf64>) outs(%320 : tensor<1x?x1xf64>) {
    ^bb0(%in: f64, %out: f64):
      %1190 = arith.divf %in, %cst_157 : f64
      linalg.yield %1190 : f64
    } -> tensor<1x?x1xf64>
    %328 = tensor.empty(%310) : tensor<1x?x1xf32>
    %329 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%327 : tensor<1x?x1xf64>) outs(%328 : tensor<1x?x1xf32>) {
    ^bb0(%in: f64, %out: f32):
      %1190 = arith.truncf %in : f64 to f32
      linalg.yield %1190 : f32
    } -> tensor<1x?x1xf32>
    %330 = linalg.fill ins(%cst_0 : f32) outs(%328 : tensor<1x?x1xf32>) -> tensor<1x?x1xf32>
    %331 = linalg.generic {indexing_maps = [#map1, #map11], iterator_types = ["parallel", "parallel", "reduction"]} ins(%317 : tensor<1x?x768xf32>) outs(%330 : tensor<1x?x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1190 = arith.addf %in, %out : f32
      linalg.yield %1190 : f32
    } -> tensor<1x?x1xf32>
    %332 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%331 : tensor<1x?x1xf32>) outs(%328 : tensor<1x?x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1190 = arith.divf %in, %cst_158 : f32
      linalg.yield %1190 : f32
    } -> tensor<1x?x1xf32>
    %333 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%329 : tensor<1x?x1xf32>) outs(%328 : tensor<1x?x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1190 = arith.truncf %cst_154 : f64 to f32
      %1191 = arith.addf %in, %1190 : f32
      linalg.yield %1191 : f32
    } -> tensor<1x?x1xf32>
    %334 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%333 : tensor<1x?x1xf32>) outs(%328 : tensor<1x?x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1190 = math.rsqrt %in : f32
      linalg.yield %1190 : f32
    } -> tensor<1x?x1xf32>
    %335 = linalg.generic {indexing_maps = [#map1, #map11, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%317, %332 : tensor<1x?x768xf32>, tensor<1x?x1xf32>) outs(%316 : tensor<1x?x768xf32>) {
    ^bb0(%in: f32, %in_495: f32, %out: f32):
      %1190 = arith.subf %in, %in_495 : f32
      linalg.yield %1190 : f32
    } -> tensor<1x?x768xf32>
    %336 = linalg.generic {indexing_maps = [#map1, #map11, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%335, %334 : tensor<1x?x768xf32>, tensor<1x?x1xf32>) outs(%316 : tensor<1x?x768xf32>) {
    ^bb0(%in: f32, %in_495: f32, %out: f32):
      %1190 = arith.mulf %in, %in_495 : f32
      linalg.yield %1190 : f32
    } -> tensor<1x?x768xf32>
    %337 = linalg.generic {indexing_maps = [#map1, #map12, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%336, %cst_119 : tensor<1x?x768xf32>, tensor<768xf32>) outs(%316 : tensor<1x?x768xf32>) {
    ^bb0(%in: f32, %in_495: f32, %out: f32):
      %1190 = arith.mulf %in, %in_495 : f32
      linalg.yield %1190 : f32
    } -> tensor<1x?x768xf32>
    %338 = linalg.generic {indexing_maps = [#map1, #map12, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%337, %cst_118 : tensor<1x?x768xf32>, tensor<768xf32>) outs(%316 : tensor<1x?x768xf32>) {
    ^bb0(%in: f32, %in_495: f32, %out: f32):
      %1190 = arith.addf %in, %in_495 : f32
      linalg.yield %1190 : f32
    } -> tensor<1x?x768xf32>
    %collapsed_243 = tensor.collapse_shape %338 [[0, 1], [2]] : tensor<1x?x768xf32> into tensor<?x768xf32>
    %339 = tensor.empty(%310) : tensor<?x3072xf32>
    %340 = linalg.fill ins(%cst_0 : f32) outs(%339 : tensor<?x3072xf32>) -> tensor<?x3072xf32>
    %341 = linalg.matmul ins(%collapsed_243, %cst_117 : tensor<?x768xf32>, tensor<768x3072xf32>) outs(%340 : tensor<?x3072xf32>) -> tensor<?x3072xf32>
    %342 = linalg.generic {indexing_maps = [#map3, #map13, #map3], iterator_types = ["parallel", "parallel"]} ins(%341, %cst_116 : tensor<?x3072xf32>, tensor<3072xf32>) outs(%339 : tensor<?x3072xf32>) {
    ^bb0(%in: f32, %in_495: f32, %out: f32):
      %1190 = arith.addf %in, %in_495 : f32
      linalg.yield %1190 : f32
    } -> tensor<?x3072xf32>
    %expanded_244 = tensor.expand_shape %342 [[0, 1], [2]] output_shape [1, %310, 3072] : tensor<?x3072xf32> into tensor<1x?x3072xf32>
    %343 = tensor.empty(%310) : tensor<1x?x3072xf32>
    %344 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%expanded_244 : tensor<1x?x3072xf32>) outs(%343 : tensor<1x?x3072xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1190 = arith.mulf %in, %cst_159 : f32
      linalg.yield %1190 : f32
    } -> tensor<1x?x3072xf32>
    %345 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%expanded_244 : tensor<1x?x3072xf32>) outs(%343 : tensor<1x?x3072xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1190 = math.fpowi %in, %c3_i64 : f32, i64
      linalg.yield %1190 : f32
    } -> tensor<1x?x3072xf32>
    %346 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%345 : tensor<1x?x3072xf32>) outs(%343 : tensor<1x?x3072xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1190 = arith.truncf %cst_141 : f64 to f32
      %1191 = arith.mulf %in, %1190 : f32
      linalg.yield %1191 : f32
    } -> tensor<1x?x3072xf32>
    %347 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%expanded_244, %346 : tensor<1x?x3072xf32>, tensor<1x?x3072xf32>) outs(%343 : tensor<1x?x3072xf32>) {
    ^bb0(%in: f32, %in_495: f32, %out: f32):
      %1190 = arith.addf %in, %in_495 : f32
      linalg.yield %1190 : f32
    } -> tensor<1x?x3072xf32>
    %348 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%347 : tensor<1x?x3072xf32>) outs(%343 : tensor<1x?x3072xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1190 = arith.truncf %cst_140 : f64 to f32
      %1191 = arith.mulf %in, %1190 : f32
      linalg.yield %1191 : f32
    } -> tensor<1x?x3072xf32>
    %349 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%348 : tensor<1x?x3072xf32>) outs(%343 : tensor<1x?x3072xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1190 = math.tanh %in : f32
      linalg.yield %1190 : f32
    } -> tensor<1x?x3072xf32>
    %350 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%349 : tensor<1x?x3072xf32>) outs(%343 : tensor<1x?x3072xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1190 = arith.addf %in, %cst_160 : f32
      linalg.yield %1190 : f32
    } -> tensor<1x?x3072xf32>
    %351 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%344, %350 : tensor<1x?x3072xf32>, tensor<1x?x3072xf32>) outs(%343 : tensor<1x?x3072xf32>) {
    ^bb0(%in: f32, %in_495: f32, %out: f32):
      %1190 = arith.mulf %in, %in_495 : f32
      linalg.yield %1190 : f32
    } -> tensor<1x?x3072xf32>
    %collapsed_245 = tensor.collapse_shape %351 [[0, 1], [2]] : tensor<1x?x3072xf32> into tensor<?x3072xf32>
    %352 = linalg.matmul ins(%collapsed_245, %cst_115 : tensor<?x3072xf32>, tensor<3072x768xf32>) outs(%312 : tensor<?x768xf32>) -> tensor<?x768xf32>
    %353 = linalg.generic {indexing_maps = [#map3, #map13, #map3], iterator_types = ["parallel", "parallel"]} ins(%352, %cst_114 : tensor<?x768xf32>, tensor<768xf32>) outs(%311 : tensor<?x768xf32>) {
    ^bb0(%in: f32, %in_495: f32, %out: f32):
      %1190 = arith.addf %in, %in_495 : f32
      linalg.yield %1190 : f32
    } -> tensor<?x768xf32>
    %expanded_246 = tensor.expand_shape %353 [[0, 1], [2]] output_shape [1, %310, 768] : tensor<?x768xf32> into tensor<1x?x768xf32>
    %354 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%317, %expanded_246 : tensor<1x?x768xf32>, tensor<1x?x768xf32>) outs(%316 : tensor<1x?x768xf32>) {
    ^bb0(%in: f32, %in_495: f32, %out: f32):
      %1190 = arith.addf %in, %in_495 : f32
      linalg.yield %1190 : f32
    } -> tensor<1x?x768xf32>
    %355 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%354 : tensor<1x?x768xf32>) outs(%318 : tensor<1x?x768xf64>) {
    ^bb0(%in: f32, %out: f64):
      %1190 = arith.extf %in : f32 to f64
      linalg.yield %1190 : f64
    } -> tensor<1x?x768xf64>
    %356 = linalg.generic {indexing_maps = [#map1, #map11], iterator_types = ["parallel", "parallel", "reduction"]} ins(%355 : tensor<1x?x768xf64>) outs(%321 : tensor<1x?x1xf64>) {
    ^bb0(%in: f64, %out: f64):
      %1190 = arith.addf %in, %out : f64
      linalg.yield %1190 : f64
    } -> tensor<1x?x1xf64>
    %357 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%356 : tensor<1x?x1xf64>) outs(%320 : tensor<1x?x1xf64>) {
    ^bb0(%in: f64, %out: f64):
      %1190 = arith.divf %in, %cst_157 : f64
      linalg.yield %1190 : f64
    } -> tensor<1x?x1xf64>
    %358 = linalg.generic {indexing_maps = [#map1, #map11, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%355, %357 : tensor<1x?x768xf64>, tensor<1x?x1xf64>) outs(%318 : tensor<1x?x768xf64>) {
    ^bb0(%in: f64, %in_495: f64, %out: f64):
      %1190 = arith.subf %in, %in_495 : f64
      linalg.yield %1190 : f64
    } -> tensor<1x?x768xf64>
    %359 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%358, %358 : tensor<1x?x768xf64>, tensor<1x?x768xf64>) outs(%318 : tensor<1x?x768xf64>) {
    ^bb0(%in: f64, %in_495: f64, %out: f64):
      %1190 = arith.mulf %in, %in_495 : f64
      linalg.yield %1190 : f64
    } -> tensor<1x?x768xf64>
    %360 = linalg.generic {indexing_maps = [#map1, #map11], iterator_types = ["parallel", "parallel", "reduction"]} ins(%359 : tensor<1x?x768xf64>) outs(%321 : tensor<1x?x1xf64>) {
    ^bb0(%in: f64, %out: f64):
      %1190 = arith.addf %in, %out : f64
      linalg.yield %1190 : f64
    } -> tensor<1x?x1xf64>
    %361 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%360 : tensor<1x?x1xf64>) outs(%320 : tensor<1x?x1xf64>) {
    ^bb0(%in: f64, %out: f64):
      %1190 = arith.divf %in, %cst_157 : f64
      linalg.yield %1190 : f64
    } -> tensor<1x?x1xf64>
    %362 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%361 : tensor<1x?x1xf64>) outs(%328 : tensor<1x?x1xf32>) {
    ^bb0(%in: f64, %out: f32):
      %1190 = arith.truncf %in : f64 to f32
      linalg.yield %1190 : f32
    } -> tensor<1x?x1xf32>
    %363 = linalg.generic {indexing_maps = [#map1, #map11], iterator_types = ["parallel", "parallel", "reduction"]} ins(%354 : tensor<1x?x768xf32>) outs(%330 : tensor<1x?x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1190 = arith.addf %in, %out : f32
      linalg.yield %1190 : f32
    } -> tensor<1x?x1xf32>
    %364 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%363 : tensor<1x?x1xf32>) outs(%328 : tensor<1x?x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1190 = arith.divf %in, %cst_158 : f32
      linalg.yield %1190 : f32
    } -> tensor<1x?x1xf32>
    %365 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%362 : tensor<1x?x1xf32>) outs(%328 : tensor<1x?x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1190 = arith.truncf %cst_154 : f64 to f32
      %1191 = arith.addf %in, %1190 : f32
      linalg.yield %1191 : f32
    } -> tensor<1x?x1xf32>
    %366 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%365 : tensor<1x?x1xf32>) outs(%328 : tensor<1x?x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1190 = math.rsqrt %in : f32
      linalg.yield %1190 : f32
    } -> tensor<1x?x1xf32>
    %367 = linalg.generic {indexing_maps = [#map1, #map11, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%354, %364 : tensor<1x?x768xf32>, tensor<1x?x1xf32>) outs(%316 : tensor<1x?x768xf32>) {
    ^bb0(%in: f32, %in_495: f32, %out: f32):
      %1190 = arith.subf %in, %in_495 : f32
      linalg.yield %1190 : f32
    } -> tensor<1x?x768xf32>
    %368 = linalg.generic {indexing_maps = [#map1, #map11, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%367, %366 : tensor<1x?x768xf32>, tensor<1x?x1xf32>) outs(%316 : tensor<1x?x768xf32>) {
    ^bb0(%in: f32, %in_495: f32, %out: f32):
      %1190 = arith.mulf %in, %in_495 : f32
      linalg.yield %1190 : f32
    } -> tensor<1x?x768xf32>
    %369 = linalg.generic {indexing_maps = [#map1, #map12, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%368, %cst_113 : tensor<1x?x768xf32>, tensor<768xf32>) outs(%316 : tensor<1x?x768xf32>) {
    ^bb0(%in: f32, %in_495: f32, %out: f32):
      %1190 = arith.mulf %in, %in_495 : f32
      linalg.yield %1190 : f32
    } -> tensor<1x?x768xf32>
    %370 = linalg.generic {indexing_maps = [#map1, #map12, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%369, %cst_112 : tensor<1x?x768xf32>, tensor<768xf32>) outs(%316 : tensor<1x?x768xf32>) {
    ^bb0(%in: f32, %in_495: f32, %out: f32):
      %1190 = arith.addf %in, %in_495 : f32
      linalg.yield %1190 : f32
    } -> tensor<1x?x768xf32>
    %collapsed_247 = tensor.collapse_shape %370 [[0, 1], [2]] : tensor<1x?x768xf32> into tensor<?x768xf32>
    %371 = tensor.empty(%310) : tensor<?x2304xf32>
    %372 = linalg.fill ins(%cst_0 : f32) outs(%371 : tensor<?x2304xf32>) -> tensor<?x2304xf32>
    %373 = linalg.matmul ins(%collapsed_247, %cst_111 : tensor<?x768xf32>, tensor<768x2304xf32>) outs(%372 : tensor<?x2304xf32>) -> tensor<?x2304xf32>
    %374 = linalg.generic {indexing_maps = [#map3, #map13, #map3], iterator_types = ["parallel", "parallel"]} ins(%373, %cst_110 : tensor<?x2304xf32>, tensor<2304xf32>) outs(%371 : tensor<?x2304xf32>) {
    ^bb0(%in: f32, %in_495: f32, %out: f32):
      %1190 = arith.addf %in, %in_495 : f32
      linalg.yield %1190 : f32
    } -> tensor<?x2304xf32>
    %expanded_248 = tensor.expand_shape %374 [[0, 1], [2]] output_shape [1, %310, 2304] : tensor<?x2304xf32> into tensor<1x?x2304xf32>
    %extracted_slice_249 = tensor.extract_slice %expanded_248[0, 0, 0] [1, %310, 768] [1, 1, 1] : tensor<1x?x2304xf32> to tensor<1x?x768xf32>
    %extracted_slice_250 = tensor.extract_slice %expanded_248[0, 0, 768] [1, %310, 768] [1, 1, 1] : tensor<1x?x2304xf32> to tensor<1x?x768xf32>
    %extracted_slice_251 = tensor.extract_slice %expanded_248[0, 0, 1536] [1, %310, 768] [1, 1, 1] : tensor<1x?x2304xf32> to tensor<1x?x768xf32>
    cf.assert %62, "must have at most one inferred (negative) dimension"
    %375 = arith.muli %129, %c768_i64 : i64
    %376 = arith.divsi %375, %61 : i64
    %377 = arith.select %1, %376, %0 : i64
    %from_elements_252 = tensor.from_elements %c1_i64, %377, %376, %c64_i64 : tensor<4xi64>
    %reshape_253 = tensor.reshape %extracted_slice_250(%from_elements_252) : (tensor<1x?x768xf32>, tensor<4xi64>) -> tensor<1x?x12x64xf32>
    %378 = arith.index_cast %377 : i64 to index
    cf.assert %62, "must have at most one inferred (negative) dimension"
    %reshape_254 = tensor.reshape %extracted_slice_251(%from_elements_252) : (tensor<1x?x768xf32>, tensor<4xi64>) -> tensor<1x?x12x64xf32>
    %379 = tensor.empty(%378) : tensor<1x12x?x64xf32>
    %transposed_255 = linalg.transpose ins(%reshape_254 : tensor<1x?x12x64xf32>) outs(%379 : tensor<1x12x?x64xf32>) permutation = [0, 2, 1, 3] 
    cf.assert %62, "must have at most one inferred (negative) dimension"
    %reshape_256 = tensor.reshape %extracted_slice_249(%from_elements_252) : (tensor<1x?x768xf32>, tensor<4xi64>) -> tensor<1x?x12x64xf32>
    %transposed_257 = linalg.transpose ins(%reshape_256 : tensor<1x?x12x64xf32>) outs(%379 : tensor<1x12x?x64xf32>) permutation = [0, 2, 1, 3] 
    %380 = linalg.generic {indexing_maps = [#map7, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%transposed_257 : tensor<1x12x?x64xf32>) outs(%379 : tensor<1x12x?x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1190 = arith.truncf %cst_148 : f64 to f32
      %1191 = arith.mulf %in, %1190 : f32
      linalg.yield %1191 : f32
    } -> tensor<1x12x?x64xf32>
    %381 = tensor.empty(%378) : tensor<1x12x64x?xf32>
    %transposed_258 = linalg.transpose ins(%reshape_253 : tensor<1x?x12x64xf32>) outs(%381 : tensor<1x12x64x?xf32>) permutation = [0, 2, 3, 1] 
    %382 = linalg.generic {indexing_maps = [#map7, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%transposed_258 : tensor<1x12x64x?xf32>) outs(%381 : tensor<1x12x64x?xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1190 = arith.truncf %cst_148 : f64 to f32
      %1191 = arith.mulf %in, %1190 : f32
      linalg.yield %1191 : f32
    } -> tensor<1x12x64x?xf32>
    cf.assert %30, "unimplemented: dynamic negative broadcast sizes"
    %383 = linalg.generic {indexing_maps = [#map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%73 : tensor<1x12x?x64xf32>) {
    ^bb0(%out: f32):
      %1190 = linalg.index 1 : index
      %1191 = linalg.index 2 : index
      %1192 = linalg.index 3 : index
      %1193 = arith.cmpi eq, %378, %c1 : index
      %1194 = arith.select %1193, %c0, %1191 : index
      %extracted = tensor.extract %380[%c0, %1190, %1194, %1192] : tensor<1x12x?x64xf32>
      linalg.yield %extracted : f32
    } -> tensor<1x12x?x64xf32>
    %collapsed_259 = tensor.collapse_shape %383 [[0, 1], [2], [3]] : tensor<1x12x?x64xf32> into tensor<12x?x64xf32>
    cf.assert %30, "unimplemented: dynamic negative broadcast sizes"
    %384 = linalg.generic {indexing_maps = [#map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%75 : tensor<1x12x64x?xf32>) {
    ^bb0(%out: f32):
      %1190 = linalg.index 1 : index
      %1191 = linalg.index 2 : index
      %1192 = linalg.index 3 : index
      %1193 = arith.cmpi eq, %378, %c1 : index
      %1194 = arith.select %1193, %c0, %1192 : index
      %extracted = tensor.extract %382[%c0, %1190, %1191, %1194] : tensor<1x12x64x?xf32>
      linalg.yield %extracted : f32
    } -> tensor<1x12x64x?xf32>
    %collapsed_260 = tensor.collapse_shape %384 [[0, 1], [2], [3]] : tensor<1x12x64x?xf32> into tensor<12x64x?xf32>
    %385 = linalg.batch_matmul ins(%collapsed_259, %collapsed_260 : tensor<12x?x64xf32>, tensor<12x64x?xf32>) outs(%78 : tensor<12x?x?xf32>) -> tensor<12x?x?xf32>
    cf.assert %85, "must have at most one inferred (negative) dimension"
    %reshape_261 = tensor.reshape %385(%from_elements_177) : (tensor<12x?x?xf32>, tensor<4xi64>) -> tensor<1x12x?x?xf32>
    cf.assert %90, "mismatched size for broadcast"
    cf.assert %90, "mismatched size for broadcast"
    %386 = linalg.generic {indexing_maps = [#map7, #map14, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%reshape_261, %69 : tensor<1x12x?x?xf32>, tensor<1x1x?x?xf32>) outs(%91 : tensor<1x12x?x?xf32>) {
    ^bb0(%in: f32, %in_495: f32, %out: f32):
      %1190 = arith.addf %in, %in_495 : f32
      linalg.yield %1190 : f32
    } -> tensor<1x12x?x?xf32>
    %387:2 = linalg.generic {indexing_maps = [#map7, #map15, #map15], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%386 : tensor<1x12x?x?xf32>) outs(%96, %94 : tensor<1x12x?xf32>, tensor<1x12x?xi64>) {
    ^bb0(%in: f32, %out: f32, %out_495: i64):
      %1190 = linalg.index 3 : index
      %1191 = arith.index_cast %1190 : index to i64
      %1192 = arith.maximumf %in, %out : f32
      %1193 = arith.cmpf ogt, %in, %out : f32
      %1194 = arith.select %1193, %1191, %out_495 : i64
      linalg.yield %1192, %1194 : f32, i64
    } -> (tensor<1x12x?xf32>, tensor<1x12x?xi64>)
    %expanded_262 = tensor.expand_shape %387#0 [[0], [1], [2, 3]] output_shape [1, 12, %89, 1] : tensor<1x12x?xf32> into tensor<1x12x?x1xf32>
    %388 = linalg.generic {indexing_maps = [#map7, #map6, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%386, %expanded_262 : tensor<1x12x?x?xf32>, tensor<1x12x?x1xf32>) outs(%91 : tensor<1x12x?x?xf32>) {
    ^bb0(%in: f32, %in_495: f32, %out: f32):
      %1190 = arith.subf %in, %in_495 : f32
      linalg.yield %1190 : f32
    } -> tensor<1x12x?x?xf32>
    %389 = linalg.generic {indexing_maps = [#map7, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%388 : tensor<1x12x?x?xf32>) outs(%91 : tensor<1x12x?x?xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1190 = math.exp %in : f32
      linalg.yield %1190 : f32
    } -> tensor<1x12x?x?xf32>
    %390 = linalg.generic {indexing_maps = [#map7, #map6], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%389 : tensor<1x12x?x?xf32>) outs(%101 : tensor<1x12x?x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1190 = arith.addf %in, %out : f32
      linalg.yield %1190 : f32
    } -> tensor<1x12x?x1xf32>
    %391 = linalg.generic {indexing_maps = [#map7, #map6, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%389, %390 : tensor<1x12x?x?xf32>, tensor<1x12x?x1xf32>) outs(%91 : tensor<1x12x?x?xf32>) {
    ^bb0(%in: f32, %in_495: f32, %out: f32):
      %1190 = arith.divf %in, %in_495 : f32
      linalg.yield %1190 : f32
    } -> tensor<1x12x?x?xf32>
    %392 = linalg.generic {indexing_maps = [#map7, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%386 : tensor<1x12x?x?xf32>) outs(%104 : tensor<1x12x?x?xi1>) {
    ^bb0(%in: f32, %out: i1):
      %1190 = arith.extf %in : f32 to f64
      %1191 = arith.cmpf oeq, %1190, %cst_149 : f64
      linalg.yield %1191 : i1
    } -> tensor<1x12x?x?xi1>
    %393 = linalg.generic {indexing_maps = [#map7, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%392 : tensor<1x12x?x?xi1>) outs(%104 : tensor<1x12x?x?xi1>) {
    ^bb0(%in: i1, %out: i1):
      %1190 = arith.cmpi eq, %in, %false : i1
      linalg.yield %1190 : i1
    } -> tensor<1x12x?x?xi1>
    %394 = linalg.generic {indexing_maps = [#map7, #map6], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%393 : tensor<1x12x?x?xi1>) outs(%108 : tensor<1x12x?x1xi1>) {
    ^bb0(%in: i1, %out: i1):
      %1190 = arith.ori %in, %out : i1
      linalg.yield %1190 : i1
    } -> tensor<1x12x?x1xi1>
    %395 = linalg.generic {indexing_maps = [#map7, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%394 : tensor<1x12x?x1xi1>) outs(%107 : tensor<1x12x?x1xi1>) {
    ^bb0(%in: i1, %out: i1):
      %1190 = arith.cmpi eq, %in, %false : i1
      linalg.yield %1190 : i1
    } -> tensor<1x12x?x1xi1>
    cf.assert %111, "negative values not allowed in new dimensions"
    cf.assert %111, "negative values not allowed in new dimensions"
    cf.assert %115, "mismatched size for broadcast"
    cf.assert %116, "mismatched size for broadcast"
    %396 = linalg.generic {indexing_maps = [#map6, #map7, #map7, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%395, %114, %391 : tensor<1x12x?x1xi1>, tensor<1x12x?x?xf32>, tensor<1x12x?x?xf32>) outs(%117 : tensor<1x12x?x?xf32>) {
    ^bb0(%in: i1, %in_495: f32, %in_496: f32, %out: f32):
      %1190 = arith.select %in, %in_495, %in_496 : f32
      linalg.yield %1190 : f32
    } -> tensor<1x12x?x?xf32>
    cf.assert %30, "unimplemented: dynamic negative broadcast sizes"
    cf.assert %30, "unimplemented: dynamic negative broadcast sizes"
    %397 = linalg.generic {indexing_maps = [#map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%119 : tensor<1x12x?x?xf32>) {
    ^bb0(%out: f32):
      %1190 = linalg.index 1 : index
      %1191 = linalg.index 2 : index
      %1192 = linalg.index 3 : index
      %1193 = arith.cmpi eq, %89, %c1 : index
      %1194 = arith.select %1193, %c0, %1191 : index
      %1195 = arith.cmpi eq, %112, %c1 : index
      %1196 = arith.select %1195, %c0, %1192 : index
      %extracted = tensor.extract %396[%c0, %1190, %1194, %1196] : tensor<1x12x?x?xf32>
      linalg.yield %extracted : f32
    } -> tensor<1x12x?x?xf32>
    cf.assert %85, "must have at most one inferred (negative) dimension"
    %reshape_263 = tensor.reshape %397(%from_elements_180) : (tensor<1x12x?x?xf32>, tensor<3xi64>) -> tensor<12x?x?xf32>
    cf.assert %30, "unimplemented: dynamic negative broadcast sizes"
    %398 = linalg.generic {indexing_maps = [#map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%73 : tensor<1x12x?x64xf32>) {
    ^bb0(%out: f32):
      %1190 = linalg.index 1 : index
      %1191 = linalg.index 2 : index
      %1192 = linalg.index 3 : index
      %1193 = arith.cmpi eq, %378, %c1 : index
      %1194 = arith.select %1193, %c0, %1191 : index
      %extracted = tensor.extract %transposed_255[%c0, %1190, %1194, %1192] : tensor<1x12x?x64xf32>
      linalg.yield %extracted : f32
    } -> tensor<1x12x?x64xf32>
    %collapsed_264 = tensor.collapse_shape %398 [[0, 1], [2], [3]] : tensor<1x12x?x64xf32> into tensor<12x?x64xf32>
    cf.assert %122, "mismatching contracting dimension"
    %399 = linalg.batch_matmul ins(%reshape_263, %collapsed_264 : tensor<12x?x?xf32>, tensor<12x?x64xf32>) outs(%124 : tensor<12x?x64xf32>) -> tensor<12x?x64xf32>
    %expanded_265 = tensor.expand_shape %399 [[0, 1], [2], [3]] output_shape [1, 12, %89, 64] : tensor<12x?x64xf32> into tensor<1x12x?x64xf32>
    %transposed_266 = linalg.transpose ins(%expanded_265 : tensor<1x12x?x64xf32>) outs(%126 : tensor<1x?x12x64xf32>) permutation = [0, 2, 1, 3] 
    cf.assert %62, "must have at most one inferred (negative) dimension"
    %reshape_267 = tensor.reshape %transposed_266(%from_elements_185) : (tensor<1x?x12x64xf32>, tensor<3xi64>) -> tensor<1x?x768xf32>
    %collapsed_268 = tensor.collapse_shape %reshape_267 [[0, 1], [2]] : tensor<1x?x768xf32> into tensor<?x768xf32>
    %400 = arith.index_cast %129 : i64 to index
    %401 = tensor.empty(%400) : tensor<?x768xf32>
    %402 = linalg.fill ins(%cst_0 : f32) outs(%401 : tensor<?x768xf32>) -> tensor<?x768xf32>
    %403 = linalg.matmul ins(%collapsed_268, %cst_109 : tensor<?x768xf32>, tensor<768x768xf32>) outs(%402 : tensor<?x768xf32>) -> tensor<?x768xf32>
    %404 = linalg.generic {indexing_maps = [#map3, #map13, #map3], iterator_types = ["parallel", "parallel"]} ins(%403, %cst_108 : tensor<?x768xf32>, tensor<768xf32>) outs(%401 : tensor<?x768xf32>) {
    ^bb0(%in: f32, %in_495: f32, %out: f32):
      %1190 = arith.addf %in, %in_495 : f32
      linalg.yield %1190 : f32
    } -> tensor<?x768xf32>
    %expanded_269 = tensor.expand_shape %404 [[0, 1], [2]] output_shape [1, %400, 768] : tensor<?x768xf32> into tensor<1x?x768xf32>
    %405 = arith.cmpi eq, %400, %310 : index
    cf.assert %405, "mismatched size for broadcast"
    %406 = tensor.empty(%400) : tensor<1x?x768xf32>
    %407 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%expanded_269, %354 : tensor<1x?x768xf32>, tensor<1x?x768xf32>) outs(%406 : tensor<1x?x768xf32>) {
    ^bb0(%in: f32, %in_495: f32, %out: f32):
      %1190 = arith.addf %in, %in_495 : f32
      linalg.yield %1190 : f32
    } -> tensor<1x?x768xf32>
    %408 = tensor.empty(%400) : tensor<1x?x768xf64>
    %409 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%407 : tensor<1x?x768xf32>) outs(%408 : tensor<1x?x768xf64>) {
    ^bb0(%in: f32, %out: f64):
      %1190 = arith.extf %in : f32 to f64
      linalg.yield %1190 : f64
    } -> tensor<1x?x768xf64>
    %410 = tensor.empty(%400) : tensor<1x?x1xf64>
    %411 = linalg.fill ins(%cst : f64) outs(%410 : tensor<1x?x1xf64>) -> tensor<1x?x1xf64>
    %412 = linalg.generic {indexing_maps = [#map1, #map11], iterator_types = ["parallel", "parallel", "reduction"]} ins(%409 : tensor<1x?x768xf64>) outs(%411 : tensor<1x?x1xf64>) {
    ^bb0(%in: f64, %out: f64):
      %1190 = arith.addf %in, %out : f64
      linalg.yield %1190 : f64
    } -> tensor<1x?x1xf64>
    %413 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%412 : tensor<1x?x1xf64>) outs(%410 : tensor<1x?x1xf64>) {
    ^bb0(%in: f64, %out: f64):
      %1190 = arith.divf %in, %cst_157 : f64
      linalg.yield %1190 : f64
    } -> tensor<1x?x1xf64>
    %414 = linalg.generic {indexing_maps = [#map1, #map11, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%409, %413 : tensor<1x?x768xf64>, tensor<1x?x1xf64>) outs(%408 : tensor<1x?x768xf64>) {
    ^bb0(%in: f64, %in_495: f64, %out: f64):
      %1190 = arith.subf %in, %in_495 : f64
      linalg.yield %1190 : f64
    } -> tensor<1x?x768xf64>
    %415 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%414, %414 : tensor<1x?x768xf64>, tensor<1x?x768xf64>) outs(%408 : tensor<1x?x768xf64>) {
    ^bb0(%in: f64, %in_495: f64, %out: f64):
      %1190 = arith.mulf %in, %in_495 : f64
      linalg.yield %1190 : f64
    } -> tensor<1x?x768xf64>
    %416 = linalg.generic {indexing_maps = [#map1, #map11], iterator_types = ["parallel", "parallel", "reduction"]} ins(%415 : tensor<1x?x768xf64>) outs(%411 : tensor<1x?x1xf64>) {
    ^bb0(%in: f64, %out: f64):
      %1190 = arith.addf %in, %out : f64
      linalg.yield %1190 : f64
    } -> tensor<1x?x1xf64>
    %417 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%416 : tensor<1x?x1xf64>) outs(%410 : tensor<1x?x1xf64>) {
    ^bb0(%in: f64, %out: f64):
      %1190 = arith.divf %in, %cst_157 : f64
      linalg.yield %1190 : f64
    } -> tensor<1x?x1xf64>
    %418 = tensor.empty(%400) : tensor<1x?x1xf32>
    %419 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%417 : tensor<1x?x1xf64>) outs(%418 : tensor<1x?x1xf32>) {
    ^bb0(%in: f64, %out: f32):
      %1190 = arith.truncf %in : f64 to f32
      linalg.yield %1190 : f32
    } -> tensor<1x?x1xf32>
    %420 = linalg.fill ins(%cst_0 : f32) outs(%418 : tensor<1x?x1xf32>) -> tensor<1x?x1xf32>
    %421 = linalg.generic {indexing_maps = [#map1, #map11], iterator_types = ["parallel", "parallel", "reduction"]} ins(%407 : tensor<1x?x768xf32>) outs(%420 : tensor<1x?x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1190 = arith.addf %in, %out : f32
      linalg.yield %1190 : f32
    } -> tensor<1x?x1xf32>
    %422 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%421 : tensor<1x?x1xf32>) outs(%418 : tensor<1x?x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1190 = arith.divf %in, %cst_158 : f32
      linalg.yield %1190 : f32
    } -> tensor<1x?x1xf32>
    %423 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%419 : tensor<1x?x1xf32>) outs(%418 : tensor<1x?x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1190 = arith.truncf %cst_154 : f64 to f32
      %1191 = arith.addf %in, %1190 : f32
      linalg.yield %1191 : f32
    } -> tensor<1x?x1xf32>
    %424 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%423 : tensor<1x?x1xf32>) outs(%418 : tensor<1x?x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1190 = math.rsqrt %in : f32
      linalg.yield %1190 : f32
    } -> tensor<1x?x1xf32>
    %425 = linalg.generic {indexing_maps = [#map1, #map11, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%407, %422 : tensor<1x?x768xf32>, tensor<1x?x1xf32>) outs(%406 : tensor<1x?x768xf32>) {
    ^bb0(%in: f32, %in_495: f32, %out: f32):
      %1190 = arith.subf %in, %in_495 : f32
      linalg.yield %1190 : f32
    } -> tensor<1x?x768xf32>
    %426 = linalg.generic {indexing_maps = [#map1, #map11, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%425, %424 : tensor<1x?x768xf32>, tensor<1x?x1xf32>) outs(%406 : tensor<1x?x768xf32>) {
    ^bb0(%in: f32, %in_495: f32, %out: f32):
      %1190 = arith.mulf %in, %in_495 : f32
      linalg.yield %1190 : f32
    } -> tensor<1x?x768xf32>
    %427 = linalg.generic {indexing_maps = [#map1, #map12, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%426, %cst_107 : tensor<1x?x768xf32>, tensor<768xf32>) outs(%406 : tensor<1x?x768xf32>) {
    ^bb0(%in: f32, %in_495: f32, %out: f32):
      %1190 = arith.mulf %in, %in_495 : f32
      linalg.yield %1190 : f32
    } -> tensor<1x?x768xf32>
    %428 = linalg.generic {indexing_maps = [#map1, #map12, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%427, %cst_106 : tensor<1x?x768xf32>, tensor<768xf32>) outs(%406 : tensor<1x?x768xf32>) {
    ^bb0(%in: f32, %in_495: f32, %out: f32):
      %1190 = arith.addf %in, %in_495 : f32
      linalg.yield %1190 : f32
    } -> tensor<1x?x768xf32>
    %collapsed_270 = tensor.collapse_shape %428 [[0, 1], [2]] : tensor<1x?x768xf32> into tensor<?x768xf32>
    %429 = tensor.empty(%400) : tensor<?x3072xf32>
    %430 = linalg.fill ins(%cst_0 : f32) outs(%429 : tensor<?x3072xf32>) -> tensor<?x3072xf32>
    %431 = linalg.matmul ins(%collapsed_270, %cst_105 : tensor<?x768xf32>, tensor<768x3072xf32>) outs(%430 : tensor<?x3072xf32>) -> tensor<?x3072xf32>
    %432 = linalg.generic {indexing_maps = [#map3, #map13, #map3], iterator_types = ["parallel", "parallel"]} ins(%431, %cst_104 : tensor<?x3072xf32>, tensor<3072xf32>) outs(%429 : tensor<?x3072xf32>) {
    ^bb0(%in: f32, %in_495: f32, %out: f32):
      %1190 = arith.addf %in, %in_495 : f32
      linalg.yield %1190 : f32
    } -> tensor<?x3072xf32>
    %expanded_271 = tensor.expand_shape %432 [[0, 1], [2]] output_shape [1, %400, 3072] : tensor<?x3072xf32> into tensor<1x?x3072xf32>
    %433 = tensor.empty(%400) : tensor<1x?x3072xf32>
    %434 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%expanded_271 : tensor<1x?x3072xf32>) outs(%433 : tensor<1x?x3072xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1190 = arith.mulf %in, %cst_159 : f32
      linalg.yield %1190 : f32
    } -> tensor<1x?x3072xf32>
    %435 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%expanded_271 : tensor<1x?x3072xf32>) outs(%433 : tensor<1x?x3072xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1190 = math.fpowi %in, %c3_i64 : f32, i64
      linalg.yield %1190 : f32
    } -> tensor<1x?x3072xf32>
    %436 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%435 : tensor<1x?x3072xf32>) outs(%433 : tensor<1x?x3072xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1190 = arith.truncf %cst_141 : f64 to f32
      %1191 = arith.mulf %in, %1190 : f32
      linalg.yield %1191 : f32
    } -> tensor<1x?x3072xf32>
    %437 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%expanded_271, %436 : tensor<1x?x3072xf32>, tensor<1x?x3072xf32>) outs(%433 : tensor<1x?x3072xf32>) {
    ^bb0(%in: f32, %in_495: f32, %out: f32):
      %1190 = arith.addf %in, %in_495 : f32
      linalg.yield %1190 : f32
    } -> tensor<1x?x3072xf32>
    %438 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%437 : tensor<1x?x3072xf32>) outs(%433 : tensor<1x?x3072xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1190 = arith.truncf %cst_140 : f64 to f32
      %1191 = arith.mulf %in, %1190 : f32
      linalg.yield %1191 : f32
    } -> tensor<1x?x3072xf32>
    %439 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%438 : tensor<1x?x3072xf32>) outs(%433 : tensor<1x?x3072xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1190 = math.tanh %in : f32
      linalg.yield %1190 : f32
    } -> tensor<1x?x3072xf32>
    %440 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%439 : tensor<1x?x3072xf32>) outs(%433 : tensor<1x?x3072xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1190 = arith.addf %in, %cst_160 : f32
      linalg.yield %1190 : f32
    } -> tensor<1x?x3072xf32>
    %441 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%434, %440 : tensor<1x?x3072xf32>, tensor<1x?x3072xf32>) outs(%433 : tensor<1x?x3072xf32>) {
    ^bb0(%in: f32, %in_495: f32, %out: f32):
      %1190 = arith.mulf %in, %in_495 : f32
      linalg.yield %1190 : f32
    } -> tensor<1x?x3072xf32>
    %collapsed_272 = tensor.collapse_shape %441 [[0, 1], [2]] : tensor<1x?x3072xf32> into tensor<?x3072xf32>
    %442 = linalg.matmul ins(%collapsed_272, %cst_103 : tensor<?x3072xf32>, tensor<3072x768xf32>) outs(%402 : tensor<?x768xf32>) -> tensor<?x768xf32>
    %443 = linalg.generic {indexing_maps = [#map3, #map13, #map3], iterator_types = ["parallel", "parallel"]} ins(%442, %cst_102 : tensor<?x768xf32>, tensor<768xf32>) outs(%401 : tensor<?x768xf32>) {
    ^bb0(%in: f32, %in_495: f32, %out: f32):
      %1190 = arith.addf %in, %in_495 : f32
      linalg.yield %1190 : f32
    } -> tensor<?x768xf32>
    %expanded_273 = tensor.expand_shape %443 [[0, 1], [2]] output_shape [1, %400, 768] : tensor<?x768xf32> into tensor<1x?x768xf32>
    %444 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%407, %expanded_273 : tensor<1x?x768xf32>, tensor<1x?x768xf32>) outs(%406 : tensor<1x?x768xf32>) {
    ^bb0(%in: f32, %in_495: f32, %out: f32):
      %1190 = arith.addf %in, %in_495 : f32
      linalg.yield %1190 : f32
    } -> tensor<1x?x768xf32>
    %445 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%444 : tensor<1x?x768xf32>) outs(%408 : tensor<1x?x768xf64>) {
    ^bb0(%in: f32, %out: f64):
      %1190 = arith.extf %in : f32 to f64
      linalg.yield %1190 : f64
    } -> tensor<1x?x768xf64>
    %446 = linalg.generic {indexing_maps = [#map1, #map11], iterator_types = ["parallel", "parallel", "reduction"]} ins(%445 : tensor<1x?x768xf64>) outs(%411 : tensor<1x?x1xf64>) {
    ^bb0(%in: f64, %out: f64):
      %1190 = arith.addf %in, %out : f64
      linalg.yield %1190 : f64
    } -> tensor<1x?x1xf64>
    %447 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%446 : tensor<1x?x1xf64>) outs(%410 : tensor<1x?x1xf64>) {
    ^bb0(%in: f64, %out: f64):
      %1190 = arith.divf %in, %cst_157 : f64
      linalg.yield %1190 : f64
    } -> tensor<1x?x1xf64>
    %448 = linalg.generic {indexing_maps = [#map1, #map11, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%445, %447 : tensor<1x?x768xf64>, tensor<1x?x1xf64>) outs(%408 : tensor<1x?x768xf64>) {
    ^bb0(%in: f64, %in_495: f64, %out: f64):
      %1190 = arith.subf %in, %in_495 : f64
      linalg.yield %1190 : f64
    } -> tensor<1x?x768xf64>
    %449 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%448, %448 : tensor<1x?x768xf64>, tensor<1x?x768xf64>) outs(%408 : tensor<1x?x768xf64>) {
    ^bb0(%in: f64, %in_495: f64, %out: f64):
      %1190 = arith.mulf %in, %in_495 : f64
      linalg.yield %1190 : f64
    } -> tensor<1x?x768xf64>
    %450 = linalg.generic {indexing_maps = [#map1, #map11], iterator_types = ["parallel", "parallel", "reduction"]} ins(%449 : tensor<1x?x768xf64>) outs(%411 : tensor<1x?x1xf64>) {
    ^bb0(%in: f64, %out: f64):
      %1190 = arith.addf %in, %out : f64
      linalg.yield %1190 : f64
    } -> tensor<1x?x1xf64>
    %451 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%450 : tensor<1x?x1xf64>) outs(%410 : tensor<1x?x1xf64>) {
    ^bb0(%in: f64, %out: f64):
      %1190 = arith.divf %in, %cst_157 : f64
      linalg.yield %1190 : f64
    } -> tensor<1x?x1xf64>
    %452 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%451 : tensor<1x?x1xf64>) outs(%418 : tensor<1x?x1xf32>) {
    ^bb0(%in: f64, %out: f32):
      %1190 = arith.truncf %in : f64 to f32
      linalg.yield %1190 : f32
    } -> tensor<1x?x1xf32>
    %453 = linalg.generic {indexing_maps = [#map1, #map11], iterator_types = ["parallel", "parallel", "reduction"]} ins(%444 : tensor<1x?x768xf32>) outs(%420 : tensor<1x?x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1190 = arith.addf %in, %out : f32
      linalg.yield %1190 : f32
    } -> tensor<1x?x1xf32>
    %454 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%453 : tensor<1x?x1xf32>) outs(%418 : tensor<1x?x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1190 = arith.divf %in, %cst_158 : f32
      linalg.yield %1190 : f32
    } -> tensor<1x?x1xf32>
    %455 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%452 : tensor<1x?x1xf32>) outs(%418 : tensor<1x?x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1190 = arith.truncf %cst_154 : f64 to f32
      %1191 = arith.addf %in, %1190 : f32
      linalg.yield %1191 : f32
    } -> tensor<1x?x1xf32>
    %456 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%455 : tensor<1x?x1xf32>) outs(%418 : tensor<1x?x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1190 = math.rsqrt %in : f32
      linalg.yield %1190 : f32
    } -> tensor<1x?x1xf32>
    %457 = linalg.generic {indexing_maps = [#map1, #map11, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%444, %454 : tensor<1x?x768xf32>, tensor<1x?x1xf32>) outs(%406 : tensor<1x?x768xf32>) {
    ^bb0(%in: f32, %in_495: f32, %out: f32):
      %1190 = arith.subf %in, %in_495 : f32
      linalg.yield %1190 : f32
    } -> tensor<1x?x768xf32>
    %458 = linalg.generic {indexing_maps = [#map1, #map11, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%457, %456 : tensor<1x?x768xf32>, tensor<1x?x1xf32>) outs(%406 : tensor<1x?x768xf32>) {
    ^bb0(%in: f32, %in_495: f32, %out: f32):
      %1190 = arith.mulf %in, %in_495 : f32
      linalg.yield %1190 : f32
    } -> tensor<1x?x768xf32>
    %459 = linalg.generic {indexing_maps = [#map1, #map12, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%458, %cst_101 : tensor<1x?x768xf32>, tensor<768xf32>) outs(%406 : tensor<1x?x768xf32>) {
    ^bb0(%in: f32, %in_495: f32, %out: f32):
      %1190 = arith.mulf %in, %in_495 : f32
      linalg.yield %1190 : f32
    } -> tensor<1x?x768xf32>
    %460 = linalg.generic {indexing_maps = [#map1, #map12, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%459, %cst_100 : tensor<1x?x768xf32>, tensor<768xf32>) outs(%406 : tensor<1x?x768xf32>) {
    ^bb0(%in: f32, %in_495: f32, %out: f32):
      %1190 = arith.addf %in, %in_495 : f32
      linalg.yield %1190 : f32
    } -> tensor<1x?x768xf32>
    %collapsed_274 = tensor.collapse_shape %460 [[0, 1], [2]] : tensor<1x?x768xf32> into tensor<?x768xf32>
    %461 = tensor.empty(%400) : tensor<?x2304xf32>
    %462 = linalg.fill ins(%cst_0 : f32) outs(%461 : tensor<?x2304xf32>) -> tensor<?x2304xf32>
    %463 = linalg.matmul ins(%collapsed_274, %cst_99 : tensor<?x768xf32>, tensor<768x2304xf32>) outs(%462 : tensor<?x2304xf32>) -> tensor<?x2304xf32>
    %464 = linalg.generic {indexing_maps = [#map3, #map13, #map3], iterator_types = ["parallel", "parallel"]} ins(%463, %cst_98 : tensor<?x2304xf32>, tensor<2304xf32>) outs(%461 : tensor<?x2304xf32>) {
    ^bb0(%in: f32, %in_495: f32, %out: f32):
      %1190 = arith.addf %in, %in_495 : f32
      linalg.yield %1190 : f32
    } -> tensor<?x2304xf32>
    %expanded_275 = tensor.expand_shape %464 [[0, 1], [2]] output_shape [1, %400, 2304] : tensor<?x2304xf32> into tensor<1x?x2304xf32>
    %extracted_slice_276 = tensor.extract_slice %expanded_275[0, 0, 0] [1, %400, 768] [1, 1, 1] : tensor<1x?x2304xf32> to tensor<1x?x768xf32>
    %extracted_slice_277 = tensor.extract_slice %expanded_275[0, 0, 768] [1, %400, 768] [1, 1, 1] : tensor<1x?x2304xf32> to tensor<1x?x768xf32>
    %extracted_slice_278 = tensor.extract_slice %expanded_275[0, 0, 1536] [1, %400, 768] [1, 1, 1] : tensor<1x?x2304xf32> to tensor<1x?x768xf32>
    cf.assert %62, "must have at most one inferred (negative) dimension"
    %465 = arith.muli %129, %c768_i64 : i64
    %466 = arith.divsi %465, %61 : i64
    %467 = arith.select %1, %466, %0 : i64
    %from_elements_279 = tensor.from_elements %c1_i64, %467, %466, %c64_i64 : tensor<4xi64>
    %reshape_280 = tensor.reshape %extracted_slice_277(%from_elements_279) : (tensor<1x?x768xf32>, tensor<4xi64>) -> tensor<1x?x12x64xf32>
    %468 = arith.index_cast %467 : i64 to index
    cf.assert %62, "must have at most one inferred (negative) dimension"
    %reshape_281 = tensor.reshape %extracted_slice_278(%from_elements_279) : (tensor<1x?x768xf32>, tensor<4xi64>) -> tensor<1x?x12x64xf32>
    %469 = tensor.empty(%468) : tensor<1x12x?x64xf32>
    %transposed_282 = linalg.transpose ins(%reshape_281 : tensor<1x?x12x64xf32>) outs(%469 : tensor<1x12x?x64xf32>) permutation = [0, 2, 1, 3] 
    cf.assert %62, "must have at most one inferred (negative) dimension"
    %reshape_283 = tensor.reshape %extracted_slice_276(%from_elements_279) : (tensor<1x?x768xf32>, tensor<4xi64>) -> tensor<1x?x12x64xf32>
    %transposed_284 = linalg.transpose ins(%reshape_283 : tensor<1x?x12x64xf32>) outs(%469 : tensor<1x12x?x64xf32>) permutation = [0, 2, 1, 3] 
    %470 = linalg.generic {indexing_maps = [#map7, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%transposed_284 : tensor<1x12x?x64xf32>) outs(%469 : tensor<1x12x?x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1190 = arith.truncf %cst_148 : f64 to f32
      %1191 = arith.mulf %in, %1190 : f32
      linalg.yield %1191 : f32
    } -> tensor<1x12x?x64xf32>
    %471 = tensor.empty(%468) : tensor<1x12x64x?xf32>
    %transposed_285 = linalg.transpose ins(%reshape_280 : tensor<1x?x12x64xf32>) outs(%471 : tensor<1x12x64x?xf32>) permutation = [0, 2, 3, 1] 
    %472 = linalg.generic {indexing_maps = [#map7, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%transposed_285 : tensor<1x12x64x?xf32>) outs(%471 : tensor<1x12x64x?xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1190 = arith.truncf %cst_148 : f64 to f32
      %1191 = arith.mulf %in, %1190 : f32
      linalg.yield %1191 : f32
    } -> tensor<1x12x64x?xf32>
    cf.assert %30, "unimplemented: dynamic negative broadcast sizes"
    %473 = linalg.generic {indexing_maps = [#map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%73 : tensor<1x12x?x64xf32>) {
    ^bb0(%out: f32):
      %1190 = linalg.index 1 : index
      %1191 = linalg.index 2 : index
      %1192 = linalg.index 3 : index
      %1193 = arith.cmpi eq, %468, %c1 : index
      %1194 = arith.select %1193, %c0, %1191 : index
      %extracted = tensor.extract %470[%c0, %1190, %1194, %1192] : tensor<1x12x?x64xf32>
      linalg.yield %extracted : f32
    } -> tensor<1x12x?x64xf32>
    %collapsed_286 = tensor.collapse_shape %473 [[0, 1], [2], [3]] : tensor<1x12x?x64xf32> into tensor<12x?x64xf32>
    cf.assert %30, "unimplemented: dynamic negative broadcast sizes"
    %474 = linalg.generic {indexing_maps = [#map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%75 : tensor<1x12x64x?xf32>) {
    ^bb0(%out: f32):
      %1190 = linalg.index 1 : index
      %1191 = linalg.index 2 : index
      %1192 = linalg.index 3 : index
      %1193 = arith.cmpi eq, %468, %c1 : index
      %1194 = arith.select %1193, %c0, %1192 : index
      %extracted = tensor.extract %472[%c0, %1190, %1191, %1194] : tensor<1x12x64x?xf32>
      linalg.yield %extracted : f32
    } -> tensor<1x12x64x?xf32>
    %collapsed_287 = tensor.collapse_shape %474 [[0, 1], [2], [3]] : tensor<1x12x64x?xf32> into tensor<12x64x?xf32>
    %475 = linalg.batch_matmul ins(%collapsed_286, %collapsed_287 : tensor<12x?x64xf32>, tensor<12x64x?xf32>) outs(%78 : tensor<12x?x?xf32>) -> tensor<12x?x?xf32>
    cf.assert %85, "must have at most one inferred (negative) dimension"
    %reshape_288 = tensor.reshape %475(%from_elements_177) : (tensor<12x?x?xf32>, tensor<4xi64>) -> tensor<1x12x?x?xf32>
    cf.assert %90, "mismatched size for broadcast"
    cf.assert %90, "mismatched size for broadcast"
    %476 = linalg.generic {indexing_maps = [#map7, #map14, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%reshape_288, %69 : tensor<1x12x?x?xf32>, tensor<1x1x?x?xf32>) outs(%91 : tensor<1x12x?x?xf32>) {
    ^bb0(%in: f32, %in_495: f32, %out: f32):
      %1190 = arith.addf %in, %in_495 : f32
      linalg.yield %1190 : f32
    } -> tensor<1x12x?x?xf32>
    %477:2 = linalg.generic {indexing_maps = [#map7, #map15, #map15], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%476 : tensor<1x12x?x?xf32>) outs(%96, %94 : tensor<1x12x?xf32>, tensor<1x12x?xi64>) {
    ^bb0(%in: f32, %out: f32, %out_495: i64):
      %1190 = linalg.index 3 : index
      %1191 = arith.index_cast %1190 : index to i64
      %1192 = arith.maximumf %in, %out : f32
      %1193 = arith.cmpf ogt, %in, %out : f32
      %1194 = arith.select %1193, %1191, %out_495 : i64
      linalg.yield %1192, %1194 : f32, i64
    } -> (tensor<1x12x?xf32>, tensor<1x12x?xi64>)
    %expanded_289 = tensor.expand_shape %477#0 [[0], [1], [2, 3]] output_shape [1, 12, %89, 1] : tensor<1x12x?xf32> into tensor<1x12x?x1xf32>
    %478 = linalg.generic {indexing_maps = [#map7, #map6, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%476, %expanded_289 : tensor<1x12x?x?xf32>, tensor<1x12x?x1xf32>) outs(%91 : tensor<1x12x?x?xf32>) {
    ^bb0(%in: f32, %in_495: f32, %out: f32):
      %1190 = arith.subf %in, %in_495 : f32
      linalg.yield %1190 : f32
    } -> tensor<1x12x?x?xf32>
    %479 = linalg.generic {indexing_maps = [#map7, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%478 : tensor<1x12x?x?xf32>) outs(%91 : tensor<1x12x?x?xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1190 = math.exp %in : f32
      linalg.yield %1190 : f32
    } -> tensor<1x12x?x?xf32>
    %480 = linalg.generic {indexing_maps = [#map7, #map6], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%479 : tensor<1x12x?x?xf32>) outs(%101 : tensor<1x12x?x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1190 = arith.addf %in, %out : f32
      linalg.yield %1190 : f32
    } -> tensor<1x12x?x1xf32>
    %481 = linalg.generic {indexing_maps = [#map7, #map6, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%479, %480 : tensor<1x12x?x?xf32>, tensor<1x12x?x1xf32>) outs(%91 : tensor<1x12x?x?xf32>) {
    ^bb0(%in: f32, %in_495: f32, %out: f32):
      %1190 = arith.divf %in, %in_495 : f32
      linalg.yield %1190 : f32
    } -> tensor<1x12x?x?xf32>
    %482 = linalg.generic {indexing_maps = [#map7, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%476 : tensor<1x12x?x?xf32>) outs(%104 : tensor<1x12x?x?xi1>) {
    ^bb0(%in: f32, %out: i1):
      %1190 = arith.extf %in : f32 to f64
      %1191 = arith.cmpf oeq, %1190, %cst_149 : f64
      linalg.yield %1191 : i1
    } -> tensor<1x12x?x?xi1>
    %483 = linalg.generic {indexing_maps = [#map7, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%482 : tensor<1x12x?x?xi1>) outs(%104 : tensor<1x12x?x?xi1>) {
    ^bb0(%in: i1, %out: i1):
      %1190 = arith.cmpi eq, %in, %false : i1
      linalg.yield %1190 : i1
    } -> tensor<1x12x?x?xi1>
    %484 = linalg.generic {indexing_maps = [#map7, #map6], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%483 : tensor<1x12x?x?xi1>) outs(%108 : tensor<1x12x?x1xi1>) {
    ^bb0(%in: i1, %out: i1):
      %1190 = arith.ori %in, %out : i1
      linalg.yield %1190 : i1
    } -> tensor<1x12x?x1xi1>
    %485 = linalg.generic {indexing_maps = [#map7, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%484 : tensor<1x12x?x1xi1>) outs(%107 : tensor<1x12x?x1xi1>) {
    ^bb0(%in: i1, %out: i1):
      %1190 = arith.cmpi eq, %in, %false : i1
      linalg.yield %1190 : i1
    } -> tensor<1x12x?x1xi1>
    cf.assert %111, "negative values not allowed in new dimensions"
    cf.assert %111, "negative values not allowed in new dimensions"
    cf.assert %115, "mismatched size for broadcast"
    cf.assert %116, "mismatched size for broadcast"
    %486 = linalg.generic {indexing_maps = [#map6, #map7, #map7, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%485, %114, %481 : tensor<1x12x?x1xi1>, tensor<1x12x?x?xf32>, tensor<1x12x?x?xf32>) outs(%117 : tensor<1x12x?x?xf32>) {
    ^bb0(%in: i1, %in_495: f32, %in_496: f32, %out: f32):
      %1190 = arith.select %in, %in_495, %in_496 : f32
      linalg.yield %1190 : f32
    } -> tensor<1x12x?x?xf32>
    cf.assert %30, "unimplemented: dynamic negative broadcast sizes"
    cf.assert %30, "unimplemented: dynamic negative broadcast sizes"
    %487 = linalg.generic {indexing_maps = [#map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%119 : tensor<1x12x?x?xf32>) {
    ^bb0(%out: f32):
      %1190 = linalg.index 1 : index
      %1191 = linalg.index 2 : index
      %1192 = linalg.index 3 : index
      %1193 = arith.cmpi eq, %89, %c1 : index
      %1194 = arith.select %1193, %c0, %1191 : index
      %1195 = arith.cmpi eq, %112, %c1 : index
      %1196 = arith.select %1195, %c0, %1192 : index
      %extracted = tensor.extract %486[%c0, %1190, %1194, %1196] : tensor<1x12x?x?xf32>
      linalg.yield %extracted : f32
    } -> tensor<1x12x?x?xf32>
    cf.assert %85, "must have at most one inferred (negative) dimension"
    %reshape_290 = tensor.reshape %487(%from_elements_180) : (tensor<1x12x?x?xf32>, tensor<3xi64>) -> tensor<12x?x?xf32>
    cf.assert %30, "unimplemented: dynamic negative broadcast sizes"
    %488 = linalg.generic {indexing_maps = [#map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%73 : tensor<1x12x?x64xf32>) {
    ^bb0(%out: f32):
      %1190 = linalg.index 1 : index
      %1191 = linalg.index 2 : index
      %1192 = linalg.index 3 : index
      %1193 = arith.cmpi eq, %468, %c1 : index
      %1194 = arith.select %1193, %c0, %1191 : index
      %extracted = tensor.extract %transposed_282[%c0, %1190, %1194, %1192] : tensor<1x12x?x64xf32>
      linalg.yield %extracted : f32
    } -> tensor<1x12x?x64xf32>
    %collapsed_291 = tensor.collapse_shape %488 [[0, 1], [2], [3]] : tensor<1x12x?x64xf32> into tensor<12x?x64xf32>
    cf.assert %122, "mismatching contracting dimension"
    %489 = linalg.batch_matmul ins(%reshape_290, %collapsed_291 : tensor<12x?x?xf32>, tensor<12x?x64xf32>) outs(%124 : tensor<12x?x64xf32>) -> tensor<12x?x64xf32>
    %expanded_292 = tensor.expand_shape %489 [[0, 1], [2], [3]] output_shape [1, 12, %89, 64] : tensor<12x?x64xf32> into tensor<1x12x?x64xf32>
    %transposed_293 = linalg.transpose ins(%expanded_292 : tensor<1x12x?x64xf32>) outs(%126 : tensor<1x?x12x64xf32>) permutation = [0, 2, 1, 3] 
    cf.assert %62, "must have at most one inferred (negative) dimension"
    %reshape_294 = tensor.reshape %transposed_293(%from_elements_185) : (tensor<1x?x12x64xf32>, tensor<3xi64>) -> tensor<1x?x768xf32>
    %collapsed_295 = tensor.collapse_shape %reshape_294 [[0, 1], [2]] : tensor<1x?x768xf32> into tensor<?x768xf32>
    %490 = arith.index_cast %129 : i64 to index
    %491 = tensor.empty(%490) : tensor<?x768xf32>
    %492 = linalg.fill ins(%cst_0 : f32) outs(%491 : tensor<?x768xf32>) -> tensor<?x768xf32>
    %493 = linalg.matmul ins(%collapsed_295, %cst_97 : tensor<?x768xf32>, tensor<768x768xf32>) outs(%492 : tensor<?x768xf32>) -> tensor<?x768xf32>
    %494 = linalg.generic {indexing_maps = [#map3, #map13, #map3], iterator_types = ["parallel", "parallel"]} ins(%493, %cst_96 : tensor<?x768xf32>, tensor<768xf32>) outs(%491 : tensor<?x768xf32>) {
    ^bb0(%in: f32, %in_495: f32, %out: f32):
      %1190 = arith.addf %in, %in_495 : f32
      linalg.yield %1190 : f32
    } -> tensor<?x768xf32>
    %expanded_296 = tensor.expand_shape %494 [[0, 1], [2]] output_shape [1, %490, 768] : tensor<?x768xf32> into tensor<1x?x768xf32>
    %495 = arith.cmpi eq, %490, %400 : index
    cf.assert %495, "mismatched size for broadcast"
    %496 = tensor.empty(%490) : tensor<1x?x768xf32>
    %497 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%expanded_296, %444 : tensor<1x?x768xf32>, tensor<1x?x768xf32>) outs(%496 : tensor<1x?x768xf32>) {
    ^bb0(%in: f32, %in_495: f32, %out: f32):
      %1190 = arith.addf %in, %in_495 : f32
      linalg.yield %1190 : f32
    } -> tensor<1x?x768xf32>
    %498 = tensor.empty(%490) : tensor<1x?x768xf64>
    %499 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%497 : tensor<1x?x768xf32>) outs(%498 : tensor<1x?x768xf64>) {
    ^bb0(%in: f32, %out: f64):
      %1190 = arith.extf %in : f32 to f64
      linalg.yield %1190 : f64
    } -> tensor<1x?x768xf64>
    %500 = tensor.empty(%490) : tensor<1x?x1xf64>
    %501 = linalg.fill ins(%cst : f64) outs(%500 : tensor<1x?x1xf64>) -> tensor<1x?x1xf64>
    %502 = linalg.generic {indexing_maps = [#map1, #map11], iterator_types = ["parallel", "parallel", "reduction"]} ins(%499 : tensor<1x?x768xf64>) outs(%501 : tensor<1x?x1xf64>) {
    ^bb0(%in: f64, %out: f64):
      %1190 = arith.addf %in, %out : f64
      linalg.yield %1190 : f64
    } -> tensor<1x?x1xf64>
    %503 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%502 : tensor<1x?x1xf64>) outs(%500 : tensor<1x?x1xf64>) {
    ^bb0(%in: f64, %out: f64):
      %1190 = arith.divf %in, %cst_157 : f64
      linalg.yield %1190 : f64
    } -> tensor<1x?x1xf64>
    %504 = linalg.generic {indexing_maps = [#map1, #map11, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%499, %503 : tensor<1x?x768xf64>, tensor<1x?x1xf64>) outs(%498 : tensor<1x?x768xf64>) {
    ^bb0(%in: f64, %in_495: f64, %out: f64):
      %1190 = arith.subf %in, %in_495 : f64
      linalg.yield %1190 : f64
    } -> tensor<1x?x768xf64>
    %505 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%504, %504 : tensor<1x?x768xf64>, tensor<1x?x768xf64>) outs(%498 : tensor<1x?x768xf64>) {
    ^bb0(%in: f64, %in_495: f64, %out: f64):
      %1190 = arith.mulf %in, %in_495 : f64
      linalg.yield %1190 : f64
    } -> tensor<1x?x768xf64>
    %506 = linalg.generic {indexing_maps = [#map1, #map11], iterator_types = ["parallel", "parallel", "reduction"]} ins(%505 : tensor<1x?x768xf64>) outs(%501 : tensor<1x?x1xf64>) {
    ^bb0(%in: f64, %out: f64):
      %1190 = arith.addf %in, %out : f64
      linalg.yield %1190 : f64
    } -> tensor<1x?x1xf64>
    %507 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%506 : tensor<1x?x1xf64>) outs(%500 : tensor<1x?x1xf64>) {
    ^bb0(%in: f64, %out: f64):
      %1190 = arith.divf %in, %cst_157 : f64
      linalg.yield %1190 : f64
    } -> tensor<1x?x1xf64>
    %508 = tensor.empty(%490) : tensor<1x?x1xf32>
    %509 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%507 : tensor<1x?x1xf64>) outs(%508 : tensor<1x?x1xf32>) {
    ^bb0(%in: f64, %out: f32):
      %1190 = arith.truncf %in : f64 to f32
      linalg.yield %1190 : f32
    } -> tensor<1x?x1xf32>
    %510 = linalg.fill ins(%cst_0 : f32) outs(%508 : tensor<1x?x1xf32>) -> tensor<1x?x1xf32>
    %511 = linalg.generic {indexing_maps = [#map1, #map11], iterator_types = ["parallel", "parallel", "reduction"]} ins(%497 : tensor<1x?x768xf32>) outs(%510 : tensor<1x?x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1190 = arith.addf %in, %out : f32
      linalg.yield %1190 : f32
    } -> tensor<1x?x1xf32>
    %512 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%511 : tensor<1x?x1xf32>) outs(%508 : tensor<1x?x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1190 = arith.divf %in, %cst_158 : f32
      linalg.yield %1190 : f32
    } -> tensor<1x?x1xf32>
    %513 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%509 : tensor<1x?x1xf32>) outs(%508 : tensor<1x?x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1190 = arith.truncf %cst_154 : f64 to f32
      %1191 = arith.addf %in, %1190 : f32
      linalg.yield %1191 : f32
    } -> tensor<1x?x1xf32>
    %514 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%513 : tensor<1x?x1xf32>) outs(%508 : tensor<1x?x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1190 = math.rsqrt %in : f32
      linalg.yield %1190 : f32
    } -> tensor<1x?x1xf32>
    %515 = linalg.generic {indexing_maps = [#map1, #map11, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%497, %512 : tensor<1x?x768xf32>, tensor<1x?x1xf32>) outs(%496 : tensor<1x?x768xf32>) {
    ^bb0(%in: f32, %in_495: f32, %out: f32):
      %1190 = arith.subf %in, %in_495 : f32
      linalg.yield %1190 : f32
    } -> tensor<1x?x768xf32>
    %516 = linalg.generic {indexing_maps = [#map1, #map11, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%515, %514 : tensor<1x?x768xf32>, tensor<1x?x1xf32>) outs(%496 : tensor<1x?x768xf32>) {
    ^bb0(%in: f32, %in_495: f32, %out: f32):
      %1190 = arith.mulf %in, %in_495 : f32
      linalg.yield %1190 : f32
    } -> tensor<1x?x768xf32>
    %517 = linalg.generic {indexing_maps = [#map1, #map12, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%516, %cst_95 : tensor<1x?x768xf32>, tensor<768xf32>) outs(%496 : tensor<1x?x768xf32>) {
    ^bb0(%in: f32, %in_495: f32, %out: f32):
      %1190 = arith.mulf %in, %in_495 : f32
      linalg.yield %1190 : f32
    } -> tensor<1x?x768xf32>
    %518 = linalg.generic {indexing_maps = [#map1, #map12, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%517, %cst_94 : tensor<1x?x768xf32>, tensor<768xf32>) outs(%496 : tensor<1x?x768xf32>) {
    ^bb0(%in: f32, %in_495: f32, %out: f32):
      %1190 = arith.addf %in, %in_495 : f32
      linalg.yield %1190 : f32
    } -> tensor<1x?x768xf32>
    %collapsed_297 = tensor.collapse_shape %518 [[0, 1], [2]] : tensor<1x?x768xf32> into tensor<?x768xf32>
    %519 = tensor.empty(%490) : tensor<?x3072xf32>
    %520 = linalg.fill ins(%cst_0 : f32) outs(%519 : tensor<?x3072xf32>) -> tensor<?x3072xf32>
    %521 = linalg.matmul ins(%collapsed_297, %cst_93 : tensor<?x768xf32>, tensor<768x3072xf32>) outs(%520 : tensor<?x3072xf32>) -> tensor<?x3072xf32>
    %522 = linalg.generic {indexing_maps = [#map3, #map13, #map3], iterator_types = ["parallel", "parallel"]} ins(%521, %cst_92 : tensor<?x3072xf32>, tensor<3072xf32>) outs(%519 : tensor<?x3072xf32>) {
    ^bb0(%in: f32, %in_495: f32, %out: f32):
      %1190 = arith.addf %in, %in_495 : f32
      linalg.yield %1190 : f32
    } -> tensor<?x3072xf32>
    %expanded_298 = tensor.expand_shape %522 [[0, 1], [2]] output_shape [1, %490, 3072] : tensor<?x3072xf32> into tensor<1x?x3072xf32>
    %523 = tensor.empty(%490) : tensor<1x?x3072xf32>
    %524 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%expanded_298 : tensor<1x?x3072xf32>) outs(%523 : tensor<1x?x3072xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1190 = arith.mulf %in, %cst_159 : f32
      linalg.yield %1190 : f32
    } -> tensor<1x?x3072xf32>
    %525 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%expanded_298 : tensor<1x?x3072xf32>) outs(%523 : tensor<1x?x3072xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1190 = math.fpowi %in, %c3_i64 : f32, i64
      linalg.yield %1190 : f32
    } -> tensor<1x?x3072xf32>
    %526 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%525 : tensor<1x?x3072xf32>) outs(%523 : tensor<1x?x3072xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1190 = arith.truncf %cst_141 : f64 to f32
      %1191 = arith.mulf %in, %1190 : f32
      linalg.yield %1191 : f32
    } -> tensor<1x?x3072xf32>
    %527 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%expanded_298, %526 : tensor<1x?x3072xf32>, tensor<1x?x3072xf32>) outs(%523 : tensor<1x?x3072xf32>) {
    ^bb0(%in: f32, %in_495: f32, %out: f32):
      %1190 = arith.addf %in, %in_495 : f32
      linalg.yield %1190 : f32
    } -> tensor<1x?x3072xf32>
    %528 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%527 : tensor<1x?x3072xf32>) outs(%523 : tensor<1x?x3072xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1190 = arith.truncf %cst_140 : f64 to f32
      %1191 = arith.mulf %in, %1190 : f32
      linalg.yield %1191 : f32
    } -> tensor<1x?x3072xf32>
    %529 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%528 : tensor<1x?x3072xf32>) outs(%523 : tensor<1x?x3072xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1190 = math.tanh %in : f32
      linalg.yield %1190 : f32
    } -> tensor<1x?x3072xf32>
    %530 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%529 : tensor<1x?x3072xf32>) outs(%523 : tensor<1x?x3072xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1190 = arith.addf %in, %cst_160 : f32
      linalg.yield %1190 : f32
    } -> tensor<1x?x3072xf32>
    %531 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%524, %530 : tensor<1x?x3072xf32>, tensor<1x?x3072xf32>) outs(%523 : tensor<1x?x3072xf32>) {
    ^bb0(%in: f32, %in_495: f32, %out: f32):
      %1190 = arith.mulf %in, %in_495 : f32
      linalg.yield %1190 : f32
    } -> tensor<1x?x3072xf32>
    %collapsed_299 = tensor.collapse_shape %531 [[0, 1], [2]] : tensor<1x?x3072xf32> into tensor<?x3072xf32>
    %532 = linalg.matmul ins(%collapsed_299, %cst_91 : tensor<?x3072xf32>, tensor<3072x768xf32>) outs(%492 : tensor<?x768xf32>) -> tensor<?x768xf32>
    %533 = linalg.generic {indexing_maps = [#map3, #map13, #map3], iterator_types = ["parallel", "parallel"]} ins(%532, %cst_90 : tensor<?x768xf32>, tensor<768xf32>) outs(%491 : tensor<?x768xf32>) {
    ^bb0(%in: f32, %in_495: f32, %out: f32):
      %1190 = arith.addf %in, %in_495 : f32
      linalg.yield %1190 : f32
    } -> tensor<?x768xf32>
    %expanded_300 = tensor.expand_shape %533 [[0, 1], [2]] output_shape [1, %490, 768] : tensor<?x768xf32> into tensor<1x?x768xf32>
    %534 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%497, %expanded_300 : tensor<1x?x768xf32>, tensor<1x?x768xf32>) outs(%496 : tensor<1x?x768xf32>) {
    ^bb0(%in: f32, %in_495: f32, %out: f32):
      %1190 = arith.addf %in, %in_495 : f32
      linalg.yield %1190 : f32
    } -> tensor<1x?x768xf32>
    %535 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%534 : tensor<1x?x768xf32>) outs(%498 : tensor<1x?x768xf64>) {
    ^bb0(%in: f32, %out: f64):
      %1190 = arith.extf %in : f32 to f64
      linalg.yield %1190 : f64
    } -> tensor<1x?x768xf64>
    %536 = linalg.generic {indexing_maps = [#map1, #map11], iterator_types = ["parallel", "parallel", "reduction"]} ins(%535 : tensor<1x?x768xf64>) outs(%501 : tensor<1x?x1xf64>) {
    ^bb0(%in: f64, %out: f64):
      %1190 = arith.addf %in, %out : f64
      linalg.yield %1190 : f64
    } -> tensor<1x?x1xf64>
    %537 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%536 : tensor<1x?x1xf64>) outs(%500 : tensor<1x?x1xf64>) {
    ^bb0(%in: f64, %out: f64):
      %1190 = arith.divf %in, %cst_157 : f64
      linalg.yield %1190 : f64
    } -> tensor<1x?x1xf64>
    %538 = linalg.generic {indexing_maps = [#map1, #map11, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%535, %537 : tensor<1x?x768xf64>, tensor<1x?x1xf64>) outs(%498 : tensor<1x?x768xf64>) {
    ^bb0(%in: f64, %in_495: f64, %out: f64):
      %1190 = arith.subf %in, %in_495 : f64
      linalg.yield %1190 : f64
    } -> tensor<1x?x768xf64>
    %539 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%538, %538 : tensor<1x?x768xf64>, tensor<1x?x768xf64>) outs(%498 : tensor<1x?x768xf64>) {
    ^bb0(%in: f64, %in_495: f64, %out: f64):
      %1190 = arith.mulf %in, %in_495 : f64
      linalg.yield %1190 : f64
    } -> tensor<1x?x768xf64>
    %540 = linalg.generic {indexing_maps = [#map1, #map11], iterator_types = ["parallel", "parallel", "reduction"]} ins(%539 : tensor<1x?x768xf64>) outs(%501 : tensor<1x?x1xf64>) {
    ^bb0(%in: f64, %out: f64):
      %1190 = arith.addf %in, %out : f64
      linalg.yield %1190 : f64
    } -> tensor<1x?x1xf64>
    %541 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%540 : tensor<1x?x1xf64>) outs(%500 : tensor<1x?x1xf64>) {
    ^bb0(%in: f64, %out: f64):
      %1190 = arith.divf %in, %cst_157 : f64
      linalg.yield %1190 : f64
    } -> tensor<1x?x1xf64>
    %542 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%541 : tensor<1x?x1xf64>) outs(%508 : tensor<1x?x1xf32>) {
    ^bb0(%in: f64, %out: f32):
      %1190 = arith.truncf %in : f64 to f32
      linalg.yield %1190 : f32
    } -> tensor<1x?x1xf32>
    %543 = linalg.generic {indexing_maps = [#map1, #map11], iterator_types = ["parallel", "parallel", "reduction"]} ins(%534 : tensor<1x?x768xf32>) outs(%510 : tensor<1x?x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1190 = arith.addf %in, %out : f32
      linalg.yield %1190 : f32
    } -> tensor<1x?x1xf32>
    %544 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%543 : tensor<1x?x1xf32>) outs(%508 : tensor<1x?x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1190 = arith.divf %in, %cst_158 : f32
      linalg.yield %1190 : f32
    } -> tensor<1x?x1xf32>
    %545 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%542 : tensor<1x?x1xf32>) outs(%508 : tensor<1x?x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1190 = arith.truncf %cst_154 : f64 to f32
      %1191 = arith.addf %in, %1190 : f32
      linalg.yield %1191 : f32
    } -> tensor<1x?x1xf32>
    %546 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%545 : tensor<1x?x1xf32>) outs(%508 : tensor<1x?x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1190 = math.rsqrt %in : f32
      linalg.yield %1190 : f32
    } -> tensor<1x?x1xf32>
    %547 = linalg.generic {indexing_maps = [#map1, #map11, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%534, %544 : tensor<1x?x768xf32>, tensor<1x?x1xf32>) outs(%496 : tensor<1x?x768xf32>) {
    ^bb0(%in: f32, %in_495: f32, %out: f32):
      %1190 = arith.subf %in, %in_495 : f32
      linalg.yield %1190 : f32
    } -> tensor<1x?x768xf32>
    %548 = linalg.generic {indexing_maps = [#map1, #map11, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%547, %546 : tensor<1x?x768xf32>, tensor<1x?x1xf32>) outs(%496 : tensor<1x?x768xf32>) {
    ^bb0(%in: f32, %in_495: f32, %out: f32):
      %1190 = arith.mulf %in, %in_495 : f32
      linalg.yield %1190 : f32
    } -> tensor<1x?x768xf32>
    %549 = linalg.generic {indexing_maps = [#map1, #map12, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%548, %cst_89 : tensor<1x?x768xf32>, tensor<768xf32>) outs(%496 : tensor<1x?x768xf32>) {
    ^bb0(%in: f32, %in_495: f32, %out: f32):
      %1190 = arith.mulf %in, %in_495 : f32
      linalg.yield %1190 : f32
    } -> tensor<1x?x768xf32>
    %550 = linalg.generic {indexing_maps = [#map1, #map12, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%549, %cst_88 : tensor<1x?x768xf32>, tensor<768xf32>) outs(%496 : tensor<1x?x768xf32>) {
    ^bb0(%in: f32, %in_495: f32, %out: f32):
      %1190 = arith.addf %in, %in_495 : f32
      linalg.yield %1190 : f32
    } -> tensor<1x?x768xf32>
    %collapsed_301 = tensor.collapse_shape %550 [[0, 1], [2]] : tensor<1x?x768xf32> into tensor<?x768xf32>
    %551 = tensor.empty(%490) : tensor<?x2304xf32>
    %552 = linalg.fill ins(%cst_0 : f32) outs(%551 : tensor<?x2304xf32>) -> tensor<?x2304xf32>
    %553 = linalg.matmul ins(%collapsed_301, %cst_87 : tensor<?x768xf32>, tensor<768x2304xf32>) outs(%552 : tensor<?x2304xf32>) -> tensor<?x2304xf32>
    %554 = linalg.generic {indexing_maps = [#map3, #map13, #map3], iterator_types = ["parallel", "parallel"]} ins(%553, %cst_86 : tensor<?x2304xf32>, tensor<2304xf32>) outs(%551 : tensor<?x2304xf32>) {
    ^bb0(%in: f32, %in_495: f32, %out: f32):
      %1190 = arith.addf %in, %in_495 : f32
      linalg.yield %1190 : f32
    } -> tensor<?x2304xf32>
    %expanded_302 = tensor.expand_shape %554 [[0, 1], [2]] output_shape [1, %490, 2304] : tensor<?x2304xf32> into tensor<1x?x2304xf32>
    %extracted_slice_303 = tensor.extract_slice %expanded_302[0, 0, 0] [1, %490, 768] [1, 1, 1] : tensor<1x?x2304xf32> to tensor<1x?x768xf32>
    %extracted_slice_304 = tensor.extract_slice %expanded_302[0, 0, 768] [1, %490, 768] [1, 1, 1] : tensor<1x?x2304xf32> to tensor<1x?x768xf32>
    %extracted_slice_305 = tensor.extract_slice %expanded_302[0, 0, 1536] [1, %490, 768] [1, 1, 1] : tensor<1x?x2304xf32> to tensor<1x?x768xf32>
    cf.assert %62, "must have at most one inferred (negative) dimension"
    %555 = arith.muli %129, %c768_i64 : i64
    %556 = arith.divsi %555, %61 : i64
    %557 = arith.select %1, %556, %0 : i64
    %from_elements_306 = tensor.from_elements %c1_i64, %557, %556, %c64_i64 : tensor<4xi64>
    %reshape_307 = tensor.reshape %extracted_slice_304(%from_elements_306) : (tensor<1x?x768xf32>, tensor<4xi64>) -> tensor<1x?x12x64xf32>
    %558 = arith.index_cast %557 : i64 to index
    cf.assert %62, "must have at most one inferred (negative) dimension"
    %reshape_308 = tensor.reshape %extracted_slice_305(%from_elements_306) : (tensor<1x?x768xf32>, tensor<4xi64>) -> tensor<1x?x12x64xf32>
    %559 = tensor.empty(%558) : tensor<1x12x?x64xf32>
    %transposed_309 = linalg.transpose ins(%reshape_308 : tensor<1x?x12x64xf32>) outs(%559 : tensor<1x12x?x64xf32>) permutation = [0, 2, 1, 3] 
    cf.assert %62, "must have at most one inferred (negative) dimension"
    %reshape_310 = tensor.reshape %extracted_slice_303(%from_elements_306) : (tensor<1x?x768xf32>, tensor<4xi64>) -> tensor<1x?x12x64xf32>
    %transposed_311 = linalg.transpose ins(%reshape_310 : tensor<1x?x12x64xf32>) outs(%559 : tensor<1x12x?x64xf32>) permutation = [0, 2, 1, 3] 
    %560 = linalg.generic {indexing_maps = [#map7, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%transposed_311 : tensor<1x12x?x64xf32>) outs(%559 : tensor<1x12x?x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1190 = arith.truncf %cst_148 : f64 to f32
      %1191 = arith.mulf %in, %1190 : f32
      linalg.yield %1191 : f32
    } -> tensor<1x12x?x64xf32>
    %561 = tensor.empty(%558) : tensor<1x12x64x?xf32>
    %transposed_312 = linalg.transpose ins(%reshape_307 : tensor<1x?x12x64xf32>) outs(%561 : tensor<1x12x64x?xf32>) permutation = [0, 2, 3, 1] 
    %562 = linalg.generic {indexing_maps = [#map7, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%transposed_312 : tensor<1x12x64x?xf32>) outs(%561 : tensor<1x12x64x?xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1190 = arith.truncf %cst_148 : f64 to f32
      %1191 = arith.mulf %in, %1190 : f32
      linalg.yield %1191 : f32
    } -> tensor<1x12x64x?xf32>
    cf.assert %30, "unimplemented: dynamic negative broadcast sizes"
    %563 = linalg.generic {indexing_maps = [#map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%73 : tensor<1x12x?x64xf32>) {
    ^bb0(%out: f32):
      %1190 = linalg.index 1 : index
      %1191 = linalg.index 2 : index
      %1192 = linalg.index 3 : index
      %1193 = arith.cmpi eq, %558, %c1 : index
      %1194 = arith.select %1193, %c0, %1191 : index
      %extracted = tensor.extract %560[%c0, %1190, %1194, %1192] : tensor<1x12x?x64xf32>
      linalg.yield %extracted : f32
    } -> tensor<1x12x?x64xf32>
    %collapsed_313 = tensor.collapse_shape %563 [[0, 1], [2], [3]] : tensor<1x12x?x64xf32> into tensor<12x?x64xf32>
    cf.assert %30, "unimplemented: dynamic negative broadcast sizes"
    %564 = linalg.generic {indexing_maps = [#map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%75 : tensor<1x12x64x?xf32>) {
    ^bb0(%out: f32):
      %1190 = linalg.index 1 : index
      %1191 = linalg.index 2 : index
      %1192 = linalg.index 3 : index
      %1193 = arith.cmpi eq, %558, %c1 : index
      %1194 = arith.select %1193, %c0, %1192 : index
      %extracted = tensor.extract %562[%c0, %1190, %1191, %1194] : tensor<1x12x64x?xf32>
      linalg.yield %extracted : f32
    } -> tensor<1x12x64x?xf32>
    %collapsed_314 = tensor.collapse_shape %564 [[0, 1], [2], [3]] : tensor<1x12x64x?xf32> into tensor<12x64x?xf32>
    %565 = linalg.batch_matmul ins(%collapsed_313, %collapsed_314 : tensor<12x?x64xf32>, tensor<12x64x?xf32>) outs(%78 : tensor<12x?x?xf32>) -> tensor<12x?x?xf32>
    cf.assert %85, "must have at most one inferred (negative) dimension"
    %reshape_315 = tensor.reshape %565(%from_elements_177) : (tensor<12x?x?xf32>, tensor<4xi64>) -> tensor<1x12x?x?xf32>
    cf.assert %90, "mismatched size for broadcast"
    cf.assert %90, "mismatched size for broadcast"
    %566 = linalg.generic {indexing_maps = [#map7, #map14, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%reshape_315, %69 : tensor<1x12x?x?xf32>, tensor<1x1x?x?xf32>) outs(%91 : tensor<1x12x?x?xf32>) {
    ^bb0(%in: f32, %in_495: f32, %out: f32):
      %1190 = arith.addf %in, %in_495 : f32
      linalg.yield %1190 : f32
    } -> tensor<1x12x?x?xf32>
    %567:2 = linalg.generic {indexing_maps = [#map7, #map15, #map15], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%566 : tensor<1x12x?x?xf32>) outs(%96, %94 : tensor<1x12x?xf32>, tensor<1x12x?xi64>) {
    ^bb0(%in: f32, %out: f32, %out_495: i64):
      %1190 = linalg.index 3 : index
      %1191 = arith.index_cast %1190 : index to i64
      %1192 = arith.maximumf %in, %out : f32
      %1193 = arith.cmpf ogt, %in, %out : f32
      %1194 = arith.select %1193, %1191, %out_495 : i64
      linalg.yield %1192, %1194 : f32, i64
    } -> (tensor<1x12x?xf32>, tensor<1x12x?xi64>)
    %expanded_316 = tensor.expand_shape %567#0 [[0], [1], [2, 3]] output_shape [1, 12, %89, 1] : tensor<1x12x?xf32> into tensor<1x12x?x1xf32>
    %568 = linalg.generic {indexing_maps = [#map7, #map6, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%566, %expanded_316 : tensor<1x12x?x?xf32>, tensor<1x12x?x1xf32>) outs(%91 : tensor<1x12x?x?xf32>) {
    ^bb0(%in: f32, %in_495: f32, %out: f32):
      %1190 = arith.subf %in, %in_495 : f32
      linalg.yield %1190 : f32
    } -> tensor<1x12x?x?xf32>
    %569 = linalg.generic {indexing_maps = [#map7, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%568 : tensor<1x12x?x?xf32>) outs(%91 : tensor<1x12x?x?xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1190 = math.exp %in : f32
      linalg.yield %1190 : f32
    } -> tensor<1x12x?x?xf32>
    %570 = linalg.generic {indexing_maps = [#map7, #map6], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%569 : tensor<1x12x?x?xf32>) outs(%101 : tensor<1x12x?x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1190 = arith.addf %in, %out : f32
      linalg.yield %1190 : f32
    } -> tensor<1x12x?x1xf32>
    %571 = linalg.generic {indexing_maps = [#map7, #map6, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%569, %570 : tensor<1x12x?x?xf32>, tensor<1x12x?x1xf32>) outs(%91 : tensor<1x12x?x?xf32>) {
    ^bb0(%in: f32, %in_495: f32, %out: f32):
      %1190 = arith.divf %in, %in_495 : f32
      linalg.yield %1190 : f32
    } -> tensor<1x12x?x?xf32>
    %572 = linalg.generic {indexing_maps = [#map7, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%566 : tensor<1x12x?x?xf32>) outs(%104 : tensor<1x12x?x?xi1>) {
    ^bb0(%in: f32, %out: i1):
      %1190 = arith.extf %in : f32 to f64
      %1191 = arith.cmpf oeq, %1190, %cst_149 : f64
      linalg.yield %1191 : i1
    } -> tensor<1x12x?x?xi1>
    %573 = linalg.generic {indexing_maps = [#map7, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%572 : tensor<1x12x?x?xi1>) outs(%104 : tensor<1x12x?x?xi1>) {
    ^bb0(%in: i1, %out: i1):
      %1190 = arith.cmpi eq, %in, %false : i1
      linalg.yield %1190 : i1
    } -> tensor<1x12x?x?xi1>
    %574 = linalg.generic {indexing_maps = [#map7, #map6], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%573 : tensor<1x12x?x?xi1>) outs(%108 : tensor<1x12x?x1xi1>) {
    ^bb0(%in: i1, %out: i1):
      %1190 = arith.ori %in, %out : i1
      linalg.yield %1190 : i1
    } -> tensor<1x12x?x1xi1>
    %575 = linalg.generic {indexing_maps = [#map7, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%574 : tensor<1x12x?x1xi1>) outs(%107 : tensor<1x12x?x1xi1>) {
    ^bb0(%in: i1, %out: i1):
      %1190 = arith.cmpi eq, %in, %false : i1
      linalg.yield %1190 : i1
    } -> tensor<1x12x?x1xi1>
    cf.assert %111, "negative values not allowed in new dimensions"
    cf.assert %111, "negative values not allowed in new dimensions"
    cf.assert %115, "mismatched size for broadcast"
    cf.assert %116, "mismatched size for broadcast"
    %576 = linalg.generic {indexing_maps = [#map6, #map7, #map7, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%575, %114, %571 : tensor<1x12x?x1xi1>, tensor<1x12x?x?xf32>, tensor<1x12x?x?xf32>) outs(%117 : tensor<1x12x?x?xf32>) {
    ^bb0(%in: i1, %in_495: f32, %in_496: f32, %out: f32):
      %1190 = arith.select %in, %in_495, %in_496 : f32
      linalg.yield %1190 : f32
    } -> tensor<1x12x?x?xf32>
    cf.assert %30, "unimplemented: dynamic negative broadcast sizes"
    cf.assert %30, "unimplemented: dynamic negative broadcast sizes"
    %577 = linalg.generic {indexing_maps = [#map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%119 : tensor<1x12x?x?xf32>) {
    ^bb0(%out: f32):
      %1190 = linalg.index 1 : index
      %1191 = linalg.index 2 : index
      %1192 = linalg.index 3 : index
      %1193 = arith.cmpi eq, %89, %c1 : index
      %1194 = arith.select %1193, %c0, %1191 : index
      %1195 = arith.cmpi eq, %112, %c1 : index
      %1196 = arith.select %1195, %c0, %1192 : index
      %extracted = tensor.extract %576[%c0, %1190, %1194, %1196] : tensor<1x12x?x?xf32>
      linalg.yield %extracted : f32
    } -> tensor<1x12x?x?xf32>
    cf.assert %85, "must have at most one inferred (negative) dimension"
    %reshape_317 = tensor.reshape %577(%from_elements_180) : (tensor<1x12x?x?xf32>, tensor<3xi64>) -> tensor<12x?x?xf32>
    cf.assert %30, "unimplemented: dynamic negative broadcast sizes"
    %578 = linalg.generic {indexing_maps = [#map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%73 : tensor<1x12x?x64xf32>) {
    ^bb0(%out: f32):
      %1190 = linalg.index 1 : index
      %1191 = linalg.index 2 : index
      %1192 = linalg.index 3 : index
      %1193 = arith.cmpi eq, %558, %c1 : index
      %1194 = arith.select %1193, %c0, %1191 : index
      %extracted = tensor.extract %transposed_309[%c0, %1190, %1194, %1192] : tensor<1x12x?x64xf32>
      linalg.yield %extracted : f32
    } -> tensor<1x12x?x64xf32>
    %collapsed_318 = tensor.collapse_shape %578 [[0, 1], [2], [3]] : tensor<1x12x?x64xf32> into tensor<12x?x64xf32>
    cf.assert %122, "mismatching contracting dimension"
    %579 = linalg.batch_matmul ins(%reshape_317, %collapsed_318 : tensor<12x?x?xf32>, tensor<12x?x64xf32>) outs(%124 : tensor<12x?x64xf32>) -> tensor<12x?x64xf32>
    %expanded_319 = tensor.expand_shape %579 [[0, 1], [2], [3]] output_shape [1, 12, %89, 64] : tensor<12x?x64xf32> into tensor<1x12x?x64xf32>
    %transposed_320 = linalg.transpose ins(%expanded_319 : tensor<1x12x?x64xf32>) outs(%126 : tensor<1x?x12x64xf32>) permutation = [0, 2, 1, 3] 
    cf.assert %62, "must have at most one inferred (negative) dimension"
    %reshape_321 = tensor.reshape %transposed_320(%from_elements_185) : (tensor<1x?x12x64xf32>, tensor<3xi64>) -> tensor<1x?x768xf32>
    %collapsed_322 = tensor.collapse_shape %reshape_321 [[0, 1], [2]] : tensor<1x?x768xf32> into tensor<?x768xf32>
    %580 = arith.index_cast %129 : i64 to index
    %581 = tensor.empty(%580) : tensor<?x768xf32>
    %582 = linalg.fill ins(%cst_0 : f32) outs(%581 : tensor<?x768xf32>) -> tensor<?x768xf32>
    %583 = linalg.matmul ins(%collapsed_322, %cst_85 : tensor<?x768xf32>, tensor<768x768xf32>) outs(%582 : tensor<?x768xf32>) -> tensor<?x768xf32>
    %584 = linalg.generic {indexing_maps = [#map3, #map13, #map3], iterator_types = ["parallel", "parallel"]} ins(%583, %cst_84 : tensor<?x768xf32>, tensor<768xf32>) outs(%581 : tensor<?x768xf32>) {
    ^bb0(%in: f32, %in_495: f32, %out: f32):
      %1190 = arith.addf %in, %in_495 : f32
      linalg.yield %1190 : f32
    } -> tensor<?x768xf32>
    %expanded_323 = tensor.expand_shape %584 [[0, 1], [2]] output_shape [1, %580, 768] : tensor<?x768xf32> into tensor<1x?x768xf32>
    %585 = arith.cmpi eq, %580, %490 : index
    cf.assert %585, "mismatched size for broadcast"
    %586 = tensor.empty(%580) : tensor<1x?x768xf32>
    %587 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%expanded_323, %534 : tensor<1x?x768xf32>, tensor<1x?x768xf32>) outs(%586 : tensor<1x?x768xf32>) {
    ^bb0(%in: f32, %in_495: f32, %out: f32):
      %1190 = arith.addf %in, %in_495 : f32
      linalg.yield %1190 : f32
    } -> tensor<1x?x768xf32>
    %588 = tensor.empty(%580) : tensor<1x?x768xf64>
    %589 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%587 : tensor<1x?x768xf32>) outs(%588 : tensor<1x?x768xf64>) {
    ^bb0(%in: f32, %out: f64):
      %1190 = arith.extf %in : f32 to f64
      linalg.yield %1190 : f64
    } -> tensor<1x?x768xf64>
    %590 = tensor.empty(%580) : tensor<1x?x1xf64>
    %591 = linalg.fill ins(%cst : f64) outs(%590 : tensor<1x?x1xf64>) -> tensor<1x?x1xf64>
    %592 = linalg.generic {indexing_maps = [#map1, #map11], iterator_types = ["parallel", "parallel", "reduction"]} ins(%589 : tensor<1x?x768xf64>) outs(%591 : tensor<1x?x1xf64>) {
    ^bb0(%in: f64, %out: f64):
      %1190 = arith.addf %in, %out : f64
      linalg.yield %1190 : f64
    } -> tensor<1x?x1xf64>
    %593 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%592 : tensor<1x?x1xf64>) outs(%590 : tensor<1x?x1xf64>) {
    ^bb0(%in: f64, %out: f64):
      %1190 = arith.divf %in, %cst_157 : f64
      linalg.yield %1190 : f64
    } -> tensor<1x?x1xf64>
    %594 = linalg.generic {indexing_maps = [#map1, #map11, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%589, %593 : tensor<1x?x768xf64>, tensor<1x?x1xf64>) outs(%588 : tensor<1x?x768xf64>) {
    ^bb0(%in: f64, %in_495: f64, %out: f64):
      %1190 = arith.subf %in, %in_495 : f64
      linalg.yield %1190 : f64
    } -> tensor<1x?x768xf64>
    %595 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%594, %594 : tensor<1x?x768xf64>, tensor<1x?x768xf64>) outs(%588 : tensor<1x?x768xf64>) {
    ^bb0(%in: f64, %in_495: f64, %out: f64):
      %1190 = arith.mulf %in, %in_495 : f64
      linalg.yield %1190 : f64
    } -> tensor<1x?x768xf64>
    %596 = linalg.generic {indexing_maps = [#map1, #map11], iterator_types = ["parallel", "parallel", "reduction"]} ins(%595 : tensor<1x?x768xf64>) outs(%591 : tensor<1x?x1xf64>) {
    ^bb0(%in: f64, %out: f64):
      %1190 = arith.addf %in, %out : f64
      linalg.yield %1190 : f64
    } -> tensor<1x?x1xf64>
    %597 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%596 : tensor<1x?x1xf64>) outs(%590 : tensor<1x?x1xf64>) {
    ^bb0(%in: f64, %out: f64):
      %1190 = arith.divf %in, %cst_157 : f64
      linalg.yield %1190 : f64
    } -> tensor<1x?x1xf64>
    %598 = tensor.empty(%580) : tensor<1x?x1xf32>
    %599 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%597 : tensor<1x?x1xf64>) outs(%598 : tensor<1x?x1xf32>) {
    ^bb0(%in: f64, %out: f32):
      %1190 = arith.truncf %in : f64 to f32
      linalg.yield %1190 : f32
    } -> tensor<1x?x1xf32>
    %600 = linalg.fill ins(%cst_0 : f32) outs(%598 : tensor<1x?x1xf32>) -> tensor<1x?x1xf32>
    %601 = linalg.generic {indexing_maps = [#map1, #map11], iterator_types = ["parallel", "parallel", "reduction"]} ins(%587 : tensor<1x?x768xf32>) outs(%600 : tensor<1x?x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1190 = arith.addf %in, %out : f32
      linalg.yield %1190 : f32
    } -> tensor<1x?x1xf32>
    %602 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%601 : tensor<1x?x1xf32>) outs(%598 : tensor<1x?x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1190 = arith.divf %in, %cst_158 : f32
      linalg.yield %1190 : f32
    } -> tensor<1x?x1xf32>
    %603 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%599 : tensor<1x?x1xf32>) outs(%598 : tensor<1x?x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1190 = arith.truncf %cst_154 : f64 to f32
      %1191 = arith.addf %in, %1190 : f32
      linalg.yield %1191 : f32
    } -> tensor<1x?x1xf32>
    %604 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%603 : tensor<1x?x1xf32>) outs(%598 : tensor<1x?x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1190 = math.rsqrt %in : f32
      linalg.yield %1190 : f32
    } -> tensor<1x?x1xf32>
    %605 = linalg.generic {indexing_maps = [#map1, #map11, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%587, %602 : tensor<1x?x768xf32>, tensor<1x?x1xf32>) outs(%586 : tensor<1x?x768xf32>) {
    ^bb0(%in: f32, %in_495: f32, %out: f32):
      %1190 = arith.subf %in, %in_495 : f32
      linalg.yield %1190 : f32
    } -> tensor<1x?x768xf32>
    %606 = linalg.generic {indexing_maps = [#map1, #map11, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%605, %604 : tensor<1x?x768xf32>, tensor<1x?x1xf32>) outs(%586 : tensor<1x?x768xf32>) {
    ^bb0(%in: f32, %in_495: f32, %out: f32):
      %1190 = arith.mulf %in, %in_495 : f32
      linalg.yield %1190 : f32
    } -> tensor<1x?x768xf32>
    %607 = linalg.generic {indexing_maps = [#map1, #map12, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%606, %cst_83 : tensor<1x?x768xf32>, tensor<768xf32>) outs(%586 : tensor<1x?x768xf32>) {
    ^bb0(%in: f32, %in_495: f32, %out: f32):
      %1190 = arith.mulf %in, %in_495 : f32
      linalg.yield %1190 : f32
    } -> tensor<1x?x768xf32>
    %608 = linalg.generic {indexing_maps = [#map1, #map12, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%607, %cst_82 : tensor<1x?x768xf32>, tensor<768xf32>) outs(%586 : tensor<1x?x768xf32>) {
    ^bb0(%in: f32, %in_495: f32, %out: f32):
      %1190 = arith.addf %in, %in_495 : f32
      linalg.yield %1190 : f32
    } -> tensor<1x?x768xf32>
    %collapsed_324 = tensor.collapse_shape %608 [[0, 1], [2]] : tensor<1x?x768xf32> into tensor<?x768xf32>
    %609 = tensor.empty(%580) : tensor<?x3072xf32>
    %610 = linalg.fill ins(%cst_0 : f32) outs(%609 : tensor<?x3072xf32>) -> tensor<?x3072xf32>
    %611 = linalg.matmul ins(%collapsed_324, %cst_81 : tensor<?x768xf32>, tensor<768x3072xf32>) outs(%610 : tensor<?x3072xf32>) -> tensor<?x3072xf32>
    %612 = linalg.generic {indexing_maps = [#map3, #map13, #map3], iterator_types = ["parallel", "parallel"]} ins(%611, %cst_80 : tensor<?x3072xf32>, tensor<3072xf32>) outs(%609 : tensor<?x3072xf32>) {
    ^bb0(%in: f32, %in_495: f32, %out: f32):
      %1190 = arith.addf %in, %in_495 : f32
      linalg.yield %1190 : f32
    } -> tensor<?x3072xf32>
    %expanded_325 = tensor.expand_shape %612 [[0, 1], [2]] output_shape [1, %580, 3072] : tensor<?x3072xf32> into tensor<1x?x3072xf32>
    %613 = tensor.empty(%580) : tensor<1x?x3072xf32>
    %614 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%expanded_325 : tensor<1x?x3072xf32>) outs(%613 : tensor<1x?x3072xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1190 = arith.mulf %in, %cst_159 : f32
      linalg.yield %1190 : f32
    } -> tensor<1x?x3072xf32>
    %615 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%expanded_325 : tensor<1x?x3072xf32>) outs(%613 : tensor<1x?x3072xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1190 = math.fpowi %in, %c3_i64 : f32, i64
      linalg.yield %1190 : f32
    } -> tensor<1x?x3072xf32>
    %616 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%615 : tensor<1x?x3072xf32>) outs(%613 : tensor<1x?x3072xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1190 = arith.truncf %cst_141 : f64 to f32
      %1191 = arith.mulf %in, %1190 : f32
      linalg.yield %1191 : f32
    } -> tensor<1x?x3072xf32>
    %617 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%expanded_325, %616 : tensor<1x?x3072xf32>, tensor<1x?x3072xf32>) outs(%613 : tensor<1x?x3072xf32>) {
    ^bb0(%in: f32, %in_495: f32, %out: f32):
      %1190 = arith.addf %in, %in_495 : f32
      linalg.yield %1190 : f32
    } -> tensor<1x?x3072xf32>
    %618 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%617 : tensor<1x?x3072xf32>) outs(%613 : tensor<1x?x3072xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1190 = arith.truncf %cst_140 : f64 to f32
      %1191 = arith.mulf %in, %1190 : f32
      linalg.yield %1191 : f32
    } -> tensor<1x?x3072xf32>
    %619 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%618 : tensor<1x?x3072xf32>) outs(%613 : tensor<1x?x3072xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1190 = math.tanh %in : f32
      linalg.yield %1190 : f32
    } -> tensor<1x?x3072xf32>
    %620 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%619 : tensor<1x?x3072xf32>) outs(%613 : tensor<1x?x3072xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1190 = arith.addf %in, %cst_160 : f32
      linalg.yield %1190 : f32
    } -> tensor<1x?x3072xf32>
    %621 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%614, %620 : tensor<1x?x3072xf32>, tensor<1x?x3072xf32>) outs(%613 : tensor<1x?x3072xf32>) {
    ^bb0(%in: f32, %in_495: f32, %out: f32):
      %1190 = arith.mulf %in, %in_495 : f32
      linalg.yield %1190 : f32
    } -> tensor<1x?x3072xf32>
    %collapsed_326 = tensor.collapse_shape %621 [[0, 1], [2]] : tensor<1x?x3072xf32> into tensor<?x3072xf32>
    %622 = linalg.matmul ins(%collapsed_326, %cst_79 : tensor<?x3072xf32>, tensor<3072x768xf32>) outs(%582 : tensor<?x768xf32>) -> tensor<?x768xf32>
    %623 = linalg.generic {indexing_maps = [#map3, #map13, #map3], iterator_types = ["parallel", "parallel"]} ins(%622, %cst_78 : tensor<?x768xf32>, tensor<768xf32>) outs(%581 : tensor<?x768xf32>) {
    ^bb0(%in: f32, %in_495: f32, %out: f32):
      %1190 = arith.addf %in, %in_495 : f32
      linalg.yield %1190 : f32
    } -> tensor<?x768xf32>
    %expanded_327 = tensor.expand_shape %623 [[0, 1], [2]] output_shape [1, %580, 768] : tensor<?x768xf32> into tensor<1x?x768xf32>
    %624 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%587, %expanded_327 : tensor<1x?x768xf32>, tensor<1x?x768xf32>) outs(%586 : tensor<1x?x768xf32>) {
    ^bb0(%in: f32, %in_495: f32, %out: f32):
      %1190 = arith.addf %in, %in_495 : f32
      linalg.yield %1190 : f32
    } -> tensor<1x?x768xf32>
    %625 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%624 : tensor<1x?x768xf32>) outs(%588 : tensor<1x?x768xf64>) {
    ^bb0(%in: f32, %out: f64):
      %1190 = arith.extf %in : f32 to f64
      linalg.yield %1190 : f64
    } -> tensor<1x?x768xf64>
    %626 = linalg.generic {indexing_maps = [#map1, #map11], iterator_types = ["parallel", "parallel", "reduction"]} ins(%625 : tensor<1x?x768xf64>) outs(%591 : tensor<1x?x1xf64>) {
    ^bb0(%in: f64, %out: f64):
      %1190 = arith.addf %in, %out : f64
      linalg.yield %1190 : f64
    } -> tensor<1x?x1xf64>
    %627 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%626 : tensor<1x?x1xf64>) outs(%590 : tensor<1x?x1xf64>) {
    ^bb0(%in: f64, %out: f64):
      %1190 = arith.divf %in, %cst_157 : f64
      linalg.yield %1190 : f64
    } -> tensor<1x?x1xf64>
    %628 = linalg.generic {indexing_maps = [#map1, #map11, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%625, %627 : tensor<1x?x768xf64>, tensor<1x?x1xf64>) outs(%588 : tensor<1x?x768xf64>) {
    ^bb0(%in: f64, %in_495: f64, %out: f64):
      %1190 = arith.subf %in, %in_495 : f64
      linalg.yield %1190 : f64
    } -> tensor<1x?x768xf64>
    %629 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%628, %628 : tensor<1x?x768xf64>, tensor<1x?x768xf64>) outs(%588 : tensor<1x?x768xf64>) {
    ^bb0(%in: f64, %in_495: f64, %out: f64):
      %1190 = arith.mulf %in, %in_495 : f64
      linalg.yield %1190 : f64
    } -> tensor<1x?x768xf64>
    %630 = linalg.generic {indexing_maps = [#map1, #map11], iterator_types = ["parallel", "parallel", "reduction"]} ins(%629 : tensor<1x?x768xf64>) outs(%591 : tensor<1x?x1xf64>) {
    ^bb0(%in: f64, %out: f64):
      %1190 = arith.addf %in, %out : f64
      linalg.yield %1190 : f64
    } -> tensor<1x?x1xf64>
    %631 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%630 : tensor<1x?x1xf64>) outs(%590 : tensor<1x?x1xf64>) {
    ^bb0(%in: f64, %out: f64):
      %1190 = arith.divf %in, %cst_157 : f64
      linalg.yield %1190 : f64
    } -> tensor<1x?x1xf64>
    %632 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%631 : tensor<1x?x1xf64>) outs(%598 : tensor<1x?x1xf32>) {
    ^bb0(%in: f64, %out: f32):
      %1190 = arith.truncf %in : f64 to f32
      linalg.yield %1190 : f32
    } -> tensor<1x?x1xf32>
    %633 = linalg.generic {indexing_maps = [#map1, #map11], iterator_types = ["parallel", "parallel", "reduction"]} ins(%624 : tensor<1x?x768xf32>) outs(%600 : tensor<1x?x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1190 = arith.addf %in, %out : f32
      linalg.yield %1190 : f32
    } -> tensor<1x?x1xf32>
    %634 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%633 : tensor<1x?x1xf32>) outs(%598 : tensor<1x?x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1190 = arith.divf %in, %cst_158 : f32
      linalg.yield %1190 : f32
    } -> tensor<1x?x1xf32>
    %635 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%632 : tensor<1x?x1xf32>) outs(%598 : tensor<1x?x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1190 = arith.truncf %cst_154 : f64 to f32
      %1191 = arith.addf %in, %1190 : f32
      linalg.yield %1191 : f32
    } -> tensor<1x?x1xf32>
    %636 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%635 : tensor<1x?x1xf32>) outs(%598 : tensor<1x?x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1190 = math.rsqrt %in : f32
      linalg.yield %1190 : f32
    } -> tensor<1x?x1xf32>
    %637 = linalg.generic {indexing_maps = [#map1, #map11, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%624, %634 : tensor<1x?x768xf32>, tensor<1x?x1xf32>) outs(%586 : tensor<1x?x768xf32>) {
    ^bb0(%in: f32, %in_495: f32, %out: f32):
      %1190 = arith.subf %in, %in_495 : f32
      linalg.yield %1190 : f32
    } -> tensor<1x?x768xf32>
    %638 = linalg.generic {indexing_maps = [#map1, #map11, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%637, %636 : tensor<1x?x768xf32>, tensor<1x?x1xf32>) outs(%586 : tensor<1x?x768xf32>) {
    ^bb0(%in: f32, %in_495: f32, %out: f32):
      %1190 = arith.mulf %in, %in_495 : f32
      linalg.yield %1190 : f32
    } -> tensor<1x?x768xf32>
    %639 = linalg.generic {indexing_maps = [#map1, #map12, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%638, %cst_77 : tensor<1x?x768xf32>, tensor<768xf32>) outs(%586 : tensor<1x?x768xf32>) {
    ^bb0(%in: f32, %in_495: f32, %out: f32):
      %1190 = arith.mulf %in, %in_495 : f32
      linalg.yield %1190 : f32
    } -> tensor<1x?x768xf32>
    %640 = linalg.generic {indexing_maps = [#map1, #map12, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%639, %cst_76 : tensor<1x?x768xf32>, tensor<768xf32>) outs(%586 : tensor<1x?x768xf32>) {
    ^bb0(%in: f32, %in_495: f32, %out: f32):
      %1190 = arith.addf %in, %in_495 : f32
      linalg.yield %1190 : f32
    } -> tensor<1x?x768xf32>
    %collapsed_328 = tensor.collapse_shape %640 [[0, 1], [2]] : tensor<1x?x768xf32> into tensor<?x768xf32>
    %641 = tensor.empty(%580) : tensor<?x2304xf32>
    %642 = linalg.fill ins(%cst_0 : f32) outs(%641 : tensor<?x2304xf32>) -> tensor<?x2304xf32>
    %643 = linalg.matmul ins(%collapsed_328, %cst_75 : tensor<?x768xf32>, tensor<768x2304xf32>) outs(%642 : tensor<?x2304xf32>) -> tensor<?x2304xf32>
    %644 = linalg.generic {indexing_maps = [#map3, #map13, #map3], iterator_types = ["parallel", "parallel"]} ins(%643, %cst_74 : tensor<?x2304xf32>, tensor<2304xf32>) outs(%641 : tensor<?x2304xf32>) {
    ^bb0(%in: f32, %in_495: f32, %out: f32):
      %1190 = arith.addf %in, %in_495 : f32
      linalg.yield %1190 : f32
    } -> tensor<?x2304xf32>
    %expanded_329 = tensor.expand_shape %644 [[0, 1], [2]] output_shape [1, %580, 2304] : tensor<?x2304xf32> into tensor<1x?x2304xf32>
    %extracted_slice_330 = tensor.extract_slice %expanded_329[0, 0, 0] [1, %580, 768] [1, 1, 1] : tensor<1x?x2304xf32> to tensor<1x?x768xf32>
    %extracted_slice_331 = tensor.extract_slice %expanded_329[0, 0, 768] [1, %580, 768] [1, 1, 1] : tensor<1x?x2304xf32> to tensor<1x?x768xf32>
    %extracted_slice_332 = tensor.extract_slice %expanded_329[0, 0, 1536] [1, %580, 768] [1, 1, 1] : tensor<1x?x2304xf32> to tensor<1x?x768xf32>
    cf.assert %62, "must have at most one inferred (negative) dimension"
    %645 = arith.muli %129, %c768_i64 : i64
    %646 = arith.divsi %645, %61 : i64
    %647 = arith.select %1, %646, %0 : i64
    %from_elements_333 = tensor.from_elements %c1_i64, %647, %646, %c64_i64 : tensor<4xi64>
    %reshape_334 = tensor.reshape %extracted_slice_331(%from_elements_333) : (tensor<1x?x768xf32>, tensor<4xi64>) -> tensor<1x?x12x64xf32>
    %648 = arith.index_cast %647 : i64 to index
    cf.assert %62, "must have at most one inferred (negative) dimension"
    %reshape_335 = tensor.reshape %extracted_slice_332(%from_elements_333) : (tensor<1x?x768xf32>, tensor<4xi64>) -> tensor<1x?x12x64xf32>
    %649 = tensor.empty(%648) : tensor<1x12x?x64xf32>
    %transposed_336 = linalg.transpose ins(%reshape_335 : tensor<1x?x12x64xf32>) outs(%649 : tensor<1x12x?x64xf32>) permutation = [0, 2, 1, 3] 
    cf.assert %62, "must have at most one inferred (negative) dimension"
    %reshape_337 = tensor.reshape %extracted_slice_330(%from_elements_333) : (tensor<1x?x768xf32>, tensor<4xi64>) -> tensor<1x?x12x64xf32>
    %transposed_338 = linalg.transpose ins(%reshape_337 : tensor<1x?x12x64xf32>) outs(%649 : tensor<1x12x?x64xf32>) permutation = [0, 2, 1, 3] 
    %650 = linalg.generic {indexing_maps = [#map7, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%transposed_338 : tensor<1x12x?x64xf32>) outs(%649 : tensor<1x12x?x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1190 = arith.truncf %cst_148 : f64 to f32
      %1191 = arith.mulf %in, %1190 : f32
      linalg.yield %1191 : f32
    } -> tensor<1x12x?x64xf32>
    %651 = tensor.empty(%648) : tensor<1x12x64x?xf32>
    %transposed_339 = linalg.transpose ins(%reshape_334 : tensor<1x?x12x64xf32>) outs(%651 : tensor<1x12x64x?xf32>) permutation = [0, 2, 3, 1] 
    %652 = linalg.generic {indexing_maps = [#map7, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%transposed_339 : tensor<1x12x64x?xf32>) outs(%651 : tensor<1x12x64x?xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1190 = arith.truncf %cst_148 : f64 to f32
      %1191 = arith.mulf %in, %1190 : f32
      linalg.yield %1191 : f32
    } -> tensor<1x12x64x?xf32>
    cf.assert %30, "unimplemented: dynamic negative broadcast sizes"
    %653 = linalg.generic {indexing_maps = [#map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%73 : tensor<1x12x?x64xf32>) {
    ^bb0(%out: f32):
      %1190 = linalg.index 1 : index
      %1191 = linalg.index 2 : index
      %1192 = linalg.index 3 : index
      %1193 = arith.cmpi eq, %648, %c1 : index
      %1194 = arith.select %1193, %c0, %1191 : index
      %extracted = tensor.extract %650[%c0, %1190, %1194, %1192] : tensor<1x12x?x64xf32>
      linalg.yield %extracted : f32
    } -> tensor<1x12x?x64xf32>
    %collapsed_340 = tensor.collapse_shape %653 [[0, 1], [2], [3]] : tensor<1x12x?x64xf32> into tensor<12x?x64xf32>
    cf.assert %30, "unimplemented: dynamic negative broadcast sizes"
    %654 = linalg.generic {indexing_maps = [#map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%75 : tensor<1x12x64x?xf32>) {
    ^bb0(%out: f32):
      %1190 = linalg.index 1 : index
      %1191 = linalg.index 2 : index
      %1192 = linalg.index 3 : index
      %1193 = arith.cmpi eq, %648, %c1 : index
      %1194 = arith.select %1193, %c0, %1192 : index
      %extracted = tensor.extract %652[%c0, %1190, %1191, %1194] : tensor<1x12x64x?xf32>
      linalg.yield %extracted : f32
    } -> tensor<1x12x64x?xf32>
    %collapsed_341 = tensor.collapse_shape %654 [[0, 1], [2], [3]] : tensor<1x12x64x?xf32> into tensor<12x64x?xf32>
    %655 = linalg.batch_matmul ins(%collapsed_340, %collapsed_341 : tensor<12x?x64xf32>, tensor<12x64x?xf32>) outs(%78 : tensor<12x?x?xf32>) -> tensor<12x?x?xf32>
    cf.assert %85, "must have at most one inferred (negative) dimension"
    %reshape_342 = tensor.reshape %655(%from_elements_177) : (tensor<12x?x?xf32>, tensor<4xi64>) -> tensor<1x12x?x?xf32>
    cf.assert %90, "mismatched size for broadcast"
    cf.assert %90, "mismatched size for broadcast"
    %656 = linalg.generic {indexing_maps = [#map7, #map14, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%reshape_342, %69 : tensor<1x12x?x?xf32>, tensor<1x1x?x?xf32>) outs(%91 : tensor<1x12x?x?xf32>) {
    ^bb0(%in: f32, %in_495: f32, %out: f32):
      %1190 = arith.addf %in, %in_495 : f32
      linalg.yield %1190 : f32
    } -> tensor<1x12x?x?xf32>
    %657:2 = linalg.generic {indexing_maps = [#map7, #map15, #map15], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%656 : tensor<1x12x?x?xf32>) outs(%96, %94 : tensor<1x12x?xf32>, tensor<1x12x?xi64>) {
    ^bb0(%in: f32, %out: f32, %out_495: i64):
      %1190 = linalg.index 3 : index
      %1191 = arith.index_cast %1190 : index to i64
      %1192 = arith.maximumf %in, %out : f32
      %1193 = arith.cmpf ogt, %in, %out : f32
      %1194 = arith.select %1193, %1191, %out_495 : i64
      linalg.yield %1192, %1194 : f32, i64
    } -> (tensor<1x12x?xf32>, tensor<1x12x?xi64>)
    %expanded_343 = tensor.expand_shape %657#0 [[0], [1], [2, 3]] output_shape [1, 12, %89, 1] : tensor<1x12x?xf32> into tensor<1x12x?x1xf32>
    %658 = linalg.generic {indexing_maps = [#map7, #map6, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%656, %expanded_343 : tensor<1x12x?x?xf32>, tensor<1x12x?x1xf32>) outs(%91 : tensor<1x12x?x?xf32>) {
    ^bb0(%in: f32, %in_495: f32, %out: f32):
      %1190 = arith.subf %in, %in_495 : f32
      linalg.yield %1190 : f32
    } -> tensor<1x12x?x?xf32>
    %659 = linalg.generic {indexing_maps = [#map7, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%658 : tensor<1x12x?x?xf32>) outs(%91 : tensor<1x12x?x?xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1190 = math.exp %in : f32
      linalg.yield %1190 : f32
    } -> tensor<1x12x?x?xf32>
    %660 = linalg.generic {indexing_maps = [#map7, #map6], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%659 : tensor<1x12x?x?xf32>) outs(%101 : tensor<1x12x?x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1190 = arith.addf %in, %out : f32
      linalg.yield %1190 : f32
    } -> tensor<1x12x?x1xf32>
    %661 = linalg.generic {indexing_maps = [#map7, #map6, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%659, %660 : tensor<1x12x?x?xf32>, tensor<1x12x?x1xf32>) outs(%91 : tensor<1x12x?x?xf32>) {
    ^bb0(%in: f32, %in_495: f32, %out: f32):
      %1190 = arith.divf %in, %in_495 : f32
      linalg.yield %1190 : f32
    } -> tensor<1x12x?x?xf32>
    %662 = linalg.generic {indexing_maps = [#map7, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%656 : tensor<1x12x?x?xf32>) outs(%104 : tensor<1x12x?x?xi1>) {
    ^bb0(%in: f32, %out: i1):
      %1190 = arith.extf %in : f32 to f64
      %1191 = arith.cmpf oeq, %1190, %cst_149 : f64
      linalg.yield %1191 : i1
    } -> tensor<1x12x?x?xi1>
    %663 = linalg.generic {indexing_maps = [#map7, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%662 : tensor<1x12x?x?xi1>) outs(%104 : tensor<1x12x?x?xi1>) {
    ^bb0(%in: i1, %out: i1):
      %1190 = arith.cmpi eq, %in, %false : i1
      linalg.yield %1190 : i1
    } -> tensor<1x12x?x?xi1>
    %664 = linalg.generic {indexing_maps = [#map7, #map6], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%663 : tensor<1x12x?x?xi1>) outs(%108 : tensor<1x12x?x1xi1>) {
    ^bb0(%in: i1, %out: i1):
      %1190 = arith.ori %in, %out : i1
      linalg.yield %1190 : i1
    } -> tensor<1x12x?x1xi1>
    %665 = linalg.generic {indexing_maps = [#map7, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%664 : tensor<1x12x?x1xi1>) outs(%107 : tensor<1x12x?x1xi1>) {
    ^bb0(%in: i1, %out: i1):
      %1190 = arith.cmpi eq, %in, %false : i1
      linalg.yield %1190 : i1
    } -> tensor<1x12x?x1xi1>
    cf.assert %111, "negative values not allowed in new dimensions"
    cf.assert %111, "negative values not allowed in new dimensions"
    cf.assert %115, "mismatched size for broadcast"
    cf.assert %116, "mismatched size for broadcast"
    %666 = linalg.generic {indexing_maps = [#map6, #map7, #map7, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%665, %114, %661 : tensor<1x12x?x1xi1>, tensor<1x12x?x?xf32>, tensor<1x12x?x?xf32>) outs(%117 : tensor<1x12x?x?xf32>) {
    ^bb0(%in: i1, %in_495: f32, %in_496: f32, %out: f32):
      %1190 = arith.select %in, %in_495, %in_496 : f32
      linalg.yield %1190 : f32
    } -> tensor<1x12x?x?xf32>
    cf.assert %30, "unimplemented: dynamic negative broadcast sizes"
    cf.assert %30, "unimplemented: dynamic negative broadcast sizes"
    %667 = linalg.generic {indexing_maps = [#map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%119 : tensor<1x12x?x?xf32>) {
    ^bb0(%out: f32):
      %1190 = linalg.index 1 : index
      %1191 = linalg.index 2 : index
      %1192 = linalg.index 3 : index
      %1193 = arith.cmpi eq, %89, %c1 : index
      %1194 = arith.select %1193, %c0, %1191 : index
      %1195 = arith.cmpi eq, %112, %c1 : index
      %1196 = arith.select %1195, %c0, %1192 : index
      %extracted = tensor.extract %666[%c0, %1190, %1194, %1196] : tensor<1x12x?x?xf32>
      linalg.yield %extracted : f32
    } -> tensor<1x12x?x?xf32>
    cf.assert %85, "must have at most one inferred (negative) dimension"
    %reshape_344 = tensor.reshape %667(%from_elements_180) : (tensor<1x12x?x?xf32>, tensor<3xi64>) -> tensor<12x?x?xf32>
    cf.assert %30, "unimplemented: dynamic negative broadcast sizes"
    %668 = linalg.generic {indexing_maps = [#map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%73 : tensor<1x12x?x64xf32>) {
    ^bb0(%out: f32):
      %1190 = linalg.index 1 : index
      %1191 = linalg.index 2 : index
      %1192 = linalg.index 3 : index
      %1193 = arith.cmpi eq, %648, %c1 : index
      %1194 = arith.select %1193, %c0, %1191 : index
      %extracted = tensor.extract %transposed_336[%c0, %1190, %1194, %1192] : tensor<1x12x?x64xf32>
      linalg.yield %extracted : f32
    } -> tensor<1x12x?x64xf32>
    %collapsed_345 = tensor.collapse_shape %668 [[0, 1], [2], [3]] : tensor<1x12x?x64xf32> into tensor<12x?x64xf32>
    cf.assert %122, "mismatching contracting dimension"
    %669 = linalg.batch_matmul ins(%reshape_344, %collapsed_345 : tensor<12x?x?xf32>, tensor<12x?x64xf32>) outs(%124 : tensor<12x?x64xf32>) -> tensor<12x?x64xf32>
    %expanded_346 = tensor.expand_shape %669 [[0, 1], [2], [3]] output_shape [1, 12, %89, 64] : tensor<12x?x64xf32> into tensor<1x12x?x64xf32>
    %transposed_347 = linalg.transpose ins(%expanded_346 : tensor<1x12x?x64xf32>) outs(%126 : tensor<1x?x12x64xf32>) permutation = [0, 2, 1, 3] 
    cf.assert %62, "must have at most one inferred (negative) dimension"
    %reshape_348 = tensor.reshape %transposed_347(%from_elements_185) : (tensor<1x?x12x64xf32>, tensor<3xi64>) -> tensor<1x?x768xf32>
    %collapsed_349 = tensor.collapse_shape %reshape_348 [[0, 1], [2]] : tensor<1x?x768xf32> into tensor<?x768xf32>
    %670 = arith.index_cast %129 : i64 to index
    %671 = tensor.empty(%670) : tensor<?x768xf32>
    %672 = linalg.fill ins(%cst_0 : f32) outs(%671 : tensor<?x768xf32>) -> tensor<?x768xf32>
    %673 = linalg.matmul ins(%collapsed_349, %cst_73 : tensor<?x768xf32>, tensor<768x768xf32>) outs(%672 : tensor<?x768xf32>) -> tensor<?x768xf32>
    %674 = linalg.generic {indexing_maps = [#map3, #map13, #map3], iterator_types = ["parallel", "parallel"]} ins(%673, %cst_72 : tensor<?x768xf32>, tensor<768xf32>) outs(%671 : tensor<?x768xf32>) {
    ^bb0(%in: f32, %in_495: f32, %out: f32):
      %1190 = arith.addf %in, %in_495 : f32
      linalg.yield %1190 : f32
    } -> tensor<?x768xf32>
    %expanded_350 = tensor.expand_shape %674 [[0, 1], [2]] output_shape [1, %670, 768] : tensor<?x768xf32> into tensor<1x?x768xf32>
    %675 = arith.cmpi eq, %670, %580 : index
    cf.assert %675, "mismatched size for broadcast"
    %676 = tensor.empty(%670) : tensor<1x?x768xf32>
    %677 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%expanded_350, %624 : tensor<1x?x768xf32>, tensor<1x?x768xf32>) outs(%676 : tensor<1x?x768xf32>) {
    ^bb0(%in: f32, %in_495: f32, %out: f32):
      %1190 = arith.addf %in, %in_495 : f32
      linalg.yield %1190 : f32
    } -> tensor<1x?x768xf32>
    %678 = tensor.empty(%670) : tensor<1x?x768xf64>
    %679 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%677 : tensor<1x?x768xf32>) outs(%678 : tensor<1x?x768xf64>) {
    ^bb0(%in: f32, %out: f64):
      %1190 = arith.extf %in : f32 to f64
      linalg.yield %1190 : f64
    } -> tensor<1x?x768xf64>
    %680 = tensor.empty(%670) : tensor<1x?x1xf64>
    %681 = linalg.fill ins(%cst : f64) outs(%680 : tensor<1x?x1xf64>) -> tensor<1x?x1xf64>
    %682 = linalg.generic {indexing_maps = [#map1, #map11], iterator_types = ["parallel", "parallel", "reduction"]} ins(%679 : tensor<1x?x768xf64>) outs(%681 : tensor<1x?x1xf64>) {
    ^bb0(%in: f64, %out: f64):
      %1190 = arith.addf %in, %out : f64
      linalg.yield %1190 : f64
    } -> tensor<1x?x1xf64>
    %683 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%682 : tensor<1x?x1xf64>) outs(%680 : tensor<1x?x1xf64>) {
    ^bb0(%in: f64, %out: f64):
      %1190 = arith.divf %in, %cst_157 : f64
      linalg.yield %1190 : f64
    } -> tensor<1x?x1xf64>
    %684 = linalg.generic {indexing_maps = [#map1, #map11, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%679, %683 : tensor<1x?x768xf64>, tensor<1x?x1xf64>) outs(%678 : tensor<1x?x768xf64>) {
    ^bb0(%in: f64, %in_495: f64, %out: f64):
      %1190 = arith.subf %in, %in_495 : f64
      linalg.yield %1190 : f64
    } -> tensor<1x?x768xf64>
    %685 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%684, %684 : tensor<1x?x768xf64>, tensor<1x?x768xf64>) outs(%678 : tensor<1x?x768xf64>) {
    ^bb0(%in: f64, %in_495: f64, %out: f64):
      %1190 = arith.mulf %in, %in_495 : f64
      linalg.yield %1190 : f64
    } -> tensor<1x?x768xf64>
    %686 = linalg.generic {indexing_maps = [#map1, #map11], iterator_types = ["parallel", "parallel", "reduction"]} ins(%685 : tensor<1x?x768xf64>) outs(%681 : tensor<1x?x1xf64>) {
    ^bb0(%in: f64, %out: f64):
      %1190 = arith.addf %in, %out : f64
      linalg.yield %1190 : f64
    } -> tensor<1x?x1xf64>
    %687 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%686 : tensor<1x?x1xf64>) outs(%680 : tensor<1x?x1xf64>) {
    ^bb0(%in: f64, %out: f64):
      %1190 = arith.divf %in, %cst_157 : f64
      linalg.yield %1190 : f64
    } -> tensor<1x?x1xf64>
    %688 = tensor.empty(%670) : tensor<1x?x1xf32>
    %689 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%687 : tensor<1x?x1xf64>) outs(%688 : tensor<1x?x1xf32>) {
    ^bb0(%in: f64, %out: f32):
      %1190 = arith.truncf %in : f64 to f32
      linalg.yield %1190 : f32
    } -> tensor<1x?x1xf32>
    %690 = linalg.fill ins(%cst_0 : f32) outs(%688 : tensor<1x?x1xf32>) -> tensor<1x?x1xf32>
    %691 = linalg.generic {indexing_maps = [#map1, #map11], iterator_types = ["parallel", "parallel", "reduction"]} ins(%677 : tensor<1x?x768xf32>) outs(%690 : tensor<1x?x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1190 = arith.addf %in, %out : f32
      linalg.yield %1190 : f32
    } -> tensor<1x?x1xf32>
    %692 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%691 : tensor<1x?x1xf32>) outs(%688 : tensor<1x?x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1190 = arith.divf %in, %cst_158 : f32
      linalg.yield %1190 : f32
    } -> tensor<1x?x1xf32>
    %693 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%689 : tensor<1x?x1xf32>) outs(%688 : tensor<1x?x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1190 = arith.truncf %cst_154 : f64 to f32
      %1191 = arith.addf %in, %1190 : f32
      linalg.yield %1191 : f32
    } -> tensor<1x?x1xf32>
    %694 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%693 : tensor<1x?x1xf32>) outs(%688 : tensor<1x?x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1190 = math.rsqrt %in : f32
      linalg.yield %1190 : f32
    } -> tensor<1x?x1xf32>
    %695 = linalg.generic {indexing_maps = [#map1, #map11, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%677, %692 : tensor<1x?x768xf32>, tensor<1x?x1xf32>) outs(%676 : tensor<1x?x768xf32>) {
    ^bb0(%in: f32, %in_495: f32, %out: f32):
      %1190 = arith.subf %in, %in_495 : f32
      linalg.yield %1190 : f32
    } -> tensor<1x?x768xf32>
    %696 = linalg.generic {indexing_maps = [#map1, #map11, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%695, %694 : tensor<1x?x768xf32>, tensor<1x?x1xf32>) outs(%676 : tensor<1x?x768xf32>) {
    ^bb0(%in: f32, %in_495: f32, %out: f32):
      %1190 = arith.mulf %in, %in_495 : f32
      linalg.yield %1190 : f32
    } -> tensor<1x?x768xf32>
    %697 = linalg.generic {indexing_maps = [#map1, #map12, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%696, %cst_71 : tensor<1x?x768xf32>, tensor<768xf32>) outs(%676 : tensor<1x?x768xf32>) {
    ^bb0(%in: f32, %in_495: f32, %out: f32):
      %1190 = arith.mulf %in, %in_495 : f32
      linalg.yield %1190 : f32
    } -> tensor<1x?x768xf32>
    %698 = linalg.generic {indexing_maps = [#map1, #map12, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%697, %cst_70 : tensor<1x?x768xf32>, tensor<768xf32>) outs(%676 : tensor<1x?x768xf32>) {
    ^bb0(%in: f32, %in_495: f32, %out: f32):
      %1190 = arith.addf %in, %in_495 : f32
      linalg.yield %1190 : f32
    } -> tensor<1x?x768xf32>
    %collapsed_351 = tensor.collapse_shape %698 [[0, 1], [2]] : tensor<1x?x768xf32> into tensor<?x768xf32>
    %699 = tensor.empty(%670) : tensor<?x3072xf32>
    %700 = linalg.fill ins(%cst_0 : f32) outs(%699 : tensor<?x3072xf32>) -> tensor<?x3072xf32>
    %701 = linalg.matmul ins(%collapsed_351, %cst_69 : tensor<?x768xf32>, tensor<768x3072xf32>) outs(%700 : tensor<?x3072xf32>) -> tensor<?x3072xf32>
    %702 = linalg.generic {indexing_maps = [#map3, #map13, #map3], iterator_types = ["parallel", "parallel"]} ins(%701, %cst_68 : tensor<?x3072xf32>, tensor<3072xf32>) outs(%699 : tensor<?x3072xf32>) {
    ^bb0(%in: f32, %in_495: f32, %out: f32):
      %1190 = arith.addf %in, %in_495 : f32
      linalg.yield %1190 : f32
    } -> tensor<?x3072xf32>
    %expanded_352 = tensor.expand_shape %702 [[0, 1], [2]] output_shape [1, %670, 3072] : tensor<?x3072xf32> into tensor<1x?x3072xf32>
    %703 = tensor.empty(%670) : tensor<1x?x3072xf32>
    %704 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%expanded_352 : tensor<1x?x3072xf32>) outs(%703 : tensor<1x?x3072xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1190 = arith.mulf %in, %cst_159 : f32
      linalg.yield %1190 : f32
    } -> tensor<1x?x3072xf32>
    %705 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%expanded_352 : tensor<1x?x3072xf32>) outs(%703 : tensor<1x?x3072xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1190 = math.fpowi %in, %c3_i64 : f32, i64
      linalg.yield %1190 : f32
    } -> tensor<1x?x3072xf32>
    %706 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%705 : tensor<1x?x3072xf32>) outs(%703 : tensor<1x?x3072xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1190 = arith.truncf %cst_141 : f64 to f32
      %1191 = arith.mulf %in, %1190 : f32
      linalg.yield %1191 : f32
    } -> tensor<1x?x3072xf32>
    %707 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%expanded_352, %706 : tensor<1x?x3072xf32>, tensor<1x?x3072xf32>) outs(%703 : tensor<1x?x3072xf32>) {
    ^bb0(%in: f32, %in_495: f32, %out: f32):
      %1190 = arith.addf %in, %in_495 : f32
      linalg.yield %1190 : f32
    } -> tensor<1x?x3072xf32>
    %708 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%707 : tensor<1x?x3072xf32>) outs(%703 : tensor<1x?x3072xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1190 = arith.truncf %cst_140 : f64 to f32
      %1191 = arith.mulf %in, %1190 : f32
      linalg.yield %1191 : f32
    } -> tensor<1x?x3072xf32>
    %709 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%708 : tensor<1x?x3072xf32>) outs(%703 : tensor<1x?x3072xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1190 = math.tanh %in : f32
      linalg.yield %1190 : f32
    } -> tensor<1x?x3072xf32>
    %710 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%709 : tensor<1x?x3072xf32>) outs(%703 : tensor<1x?x3072xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1190 = arith.addf %in, %cst_160 : f32
      linalg.yield %1190 : f32
    } -> tensor<1x?x3072xf32>
    %711 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%704, %710 : tensor<1x?x3072xf32>, tensor<1x?x3072xf32>) outs(%703 : tensor<1x?x3072xf32>) {
    ^bb0(%in: f32, %in_495: f32, %out: f32):
      %1190 = arith.mulf %in, %in_495 : f32
      linalg.yield %1190 : f32
    } -> tensor<1x?x3072xf32>
    %collapsed_353 = tensor.collapse_shape %711 [[0, 1], [2]] : tensor<1x?x3072xf32> into tensor<?x3072xf32>
    %712 = linalg.matmul ins(%collapsed_353, %cst_67 : tensor<?x3072xf32>, tensor<3072x768xf32>) outs(%672 : tensor<?x768xf32>) -> tensor<?x768xf32>
    %713 = linalg.generic {indexing_maps = [#map3, #map13, #map3], iterator_types = ["parallel", "parallel"]} ins(%712, %cst_66 : tensor<?x768xf32>, tensor<768xf32>) outs(%671 : tensor<?x768xf32>) {
    ^bb0(%in: f32, %in_495: f32, %out: f32):
      %1190 = arith.addf %in, %in_495 : f32
      linalg.yield %1190 : f32
    } -> tensor<?x768xf32>
    %expanded_354 = tensor.expand_shape %713 [[0, 1], [2]] output_shape [1, %670, 768] : tensor<?x768xf32> into tensor<1x?x768xf32>
    %714 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%677, %expanded_354 : tensor<1x?x768xf32>, tensor<1x?x768xf32>) outs(%676 : tensor<1x?x768xf32>) {
    ^bb0(%in: f32, %in_495: f32, %out: f32):
      %1190 = arith.addf %in, %in_495 : f32
      linalg.yield %1190 : f32
    } -> tensor<1x?x768xf32>
    %715 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%714 : tensor<1x?x768xf32>) outs(%678 : tensor<1x?x768xf64>) {
    ^bb0(%in: f32, %out: f64):
      %1190 = arith.extf %in : f32 to f64
      linalg.yield %1190 : f64
    } -> tensor<1x?x768xf64>
    %716 = linalg.generic {indexing_maps = [#map1, #map11], iterator_types = ["parallel", "parallel", "reduction"]} ins(%715 : tensor<1x?x768xf64>) outs(%681 : tensor<1x?x1xf64>) {
    ^bb0(%in: f64, %out: f64):
      %1190 = arith.addf %in, %out : f64
      linalg.yield %1190 : f64
    } -> tensor<1x?x1xf64>
    %717 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%716 : tensor<1x?x1xf64>) outs(%680 : tensor<1x?x1xf64>) {
    ^bb0(%in: f64, %out: f64):
      %1190 = arith.divf %in, %cst_157 : f64
      linalg.yield %1190 : f64
    } -> tensor<1x?x1xf64>
    %718 = linalg.generic {indexing_maps = [#map1, #map11, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%715, %717 : tensor<1x?x768xf64>, tensor<1x?x1xf64>) outs(%678 : tensor<1x?x768xf64>) {
    ^bb0(%in: f64, %in_495: f64, %out: f64):
      %1190 = arith.subf %in, %in_495 : f64
      linalg.yield %1190 : f64
    } -> tensor<1x?x768xf64>
    %719 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%718, %718 : tensor<1x?x768xf64>, tensor<1x?x768xf64>) outs(%678 : tensor<1x?x768xf64>) {
    ^bb0(%in: f64, %in_495: f64, %out: f64):
      %1190 = arith.mulf %in, %in_495 : f64
      linalg.yield %1190 : f64
    } -> tensor<1x?x768xf64>
    %720 = linalg.generic {indexing_maps = [#map1, #map11], iterator_types = ["parallel", "parallel", "reduction"]} ins(%719 : tensor<1x?x768xf64>) outs(%681 : tensor<1x?x1xf64>) {
    ^bb0(%in: f64, %out: f64):
      %1190 = arith.addf %in, %out : f64
      linalg.yield %1190 : f64
    } -> tensor<1x?x1xf64>
    %721 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%720 : tensor<1x?x1xf64>) outs(%680 : tensor<1x?x1xf64>) {
    ^bb0(%in: f64, %out: f64):
      %1190 = arith.divf %in, %cst_157 : f64
      linalg.yield %1190 : f64
    } -> tensor<1x?x1xf64>
    %722 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%721 : tensor<1x?x1xf64>) outs(%688 : tensor<1x?x1xf32>) {
    ^bb0(%in: f64, %out: f32):
      %1190 = arith.truncf %in : f64 to f32
      linalg.yield %1190 : f32
    } -> tensor<1x?x1xf32>
    %723 = linalg.generic {indexing_maps = [#map1, #map11], iterator_types = ["parallel", "parallel", "reduction"]} ins(%714 : tensor<1x?x768xf32>) outs(%690 : tensor<1x?x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1190 = arith.addf %in, %out : f32
      linalg.yield %1190 : f32
    } -> tensor<1x?x1xf32>
    %724 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%723 : tensor<1x?x1xf32>) outs(%688 : tensor<1x?x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1190 = arith.divf %in, %cst_158 : f32
      linalg.yield %1190 : f32
    } -> tensor<1x?x1xf32>
    %725 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%722 : tensor<1x?x1xf32>) outs(%688 : tensor<1x?x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1190 = arith.truncf %cst_154 : f64 to f32
      %1191 = arith.addf %in, %1190 : f32
      linalg.yield %1191 : f32
    } -> tensor<1x?x1xf32>
    %726 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%725 : tensor<1x?x1xf32>) outs(%688 : tensor<1x?x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1190 = math.rsqrt %in : f32
      linalg.yield %1190 : f32
    } -> tensor<1x?x1xf32>
    %727 = linalg.generic {indexing_maps = [#map1, #map11, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%714, %724 : tensor<1x?x768xf32>, tensor<1x?x1xf32>) outs(%676 : tensor<1x?x768xf32>) {
    ^bb0(%in: f32, %in_495: f32, %out: f32):
      %1190 = arith.subf %in, %in_495 : f32
      linalg.yield %1190 : f32
    } -> tensor<1x?x768xf32>
    %728 = linalg.generic {indexing_maps = [#map1, #map11, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%727, %726 : tensor<1x?x768xf32>, tensor<1x?x1xf32>) outs(%676 : tensor<1x?x768xf32>) {
    ^bb0(%in: f32, %in_495: f32, %out: f32):
      %1190 = arith.mulf %in, %in_495 : f32
      linalg.yield %1190 : f32
    } -> tensor<1x?x768xf32>
    %729 = linalg.generic {indexing_maps = [#map1, #map12, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%728, %cst_65 : tensor<1x?x768xf32>, tensor<768xf32>) outs(%676 : tensor<1x?x768xf32>) {
    ^bb0(%in: f32, %in_495: f32, %out: f32):
      %1190 = arith.mulf %in, %in_495 : f32
      linalg.yield %1190 : f32
    } -> tensor<1x?x768xf32>
    %730 = linalg.generic {indexing_maps = [#map1, #map12, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%729, %cst_64 : tensor<1x?x768xf32>, tensor<768xf32>) outs(%676 : tensor<1x?x768xf32>) {
    ^bb0(%in: f32, %in_495: f32, %out: f32):
      %1190 = arith.addf %in, %in_495 : f32
      linalg.yield %1190 : f32
    } -> tensor<1x?x768xf32>
    %collapsed_355 = tensor.collapse_shape %730 [[0, 1], [2]] : tensor<1x?x768xf32> into tensor<?x768xf32>
    %731 = tensor.empty(%670) : tensor<?x2304xf32>
    %732 = linalg.fill ins(%cst_0 : f32) outs(%731 : tensor<?x2304xf32>) -> tensor<?x2304xf32>
    %733 = linalg.matmul ins(%collapsed_355, %cst_63 : tensor<?x768xf32>, tensor<768x2304xf32>) outs(%732 : tensor<?x2304xf32>) -> tensor<?x2304xf32>
    %734 = linalg.generic {indexing_maps = [#map3, #map13, #map3], iterator_types = ["parallel", "parallel"]} ins(%733, %cst_62 : tensor<?x2304xf32>, tensor<2304xf32>) outs(%731 : tensor<?x2304xf32>) {
    ^bb0(%in: f32, %in_495: f32, %out: f32):
      %1190 = arith.addf %in, %in_495 : f32
      linalg.yield %1190 : f32
    } -> tensor<?x2304xf32>
    %expanded_356 = tensor.expand_shape %734 [[0, 1], [2]] output_shape [1, %670, 2304] : tensor<?x2304xf32> into tensor<1x?x2304xf32>
    %extracted_slice_357 = tensor.extract_slice %expanded_356[0, 0, 0] [1, %670, 768] [1, 1, 1] : tensor<1x?x2304xf32> to tensor<1x?x768xf32>
    %extracted_slice_358 = tensor.extract_slice %expanded_356[0, 0, 768] [1, %670, 768] [1, 1, 1] : tensor<1x?x2304xf32> to tensor<1x?x768xf32>
    %extracted_slice_359 = tensor.extract_slice %expanded_356[0, 0, 1536] [1, %670, 768] [1, 1, 1] : tensor<1x?x2304xf32> to tensor<1x?x768xf32>
    cf.assert %62, "must have at most one inferred (negative) dimension"
    %735 = arith.muli %129, %c768_i64 : i64
    %736 = arith.divsi %735, %61 : i64
    %737 = arith.select %1, %736, %0 : i64
    %from_elements_360 = tensor.from_elements %c1_i64, %737, %736, %c64_i64 : tensor<4xi64>
    %reshape_361 = tensor.reshape %extracted_slice_358(%from_elements_360) : (tensor<1x?x768xf32>, tensor<4xi64>) -> tensor<1x?x12x64xf32>
    %738 = arith.index_cast %737 : i64 to index
    cf.assert %62, "must have at most one inferred (negative) dimension"
    %reshape_362 = tensor.reshape %extracted_slice_359(%from_elements_360) : (tensor<1x?x768xf32>, tensor<4xi64>) -> tensor<1x?x12x64xf32>
    %739 = tensor.empty(%738) : tensor<1x12x?x64xf32>
    %transposed_363 = linalg.transpose ins(%reshape_362 : tensor<1x?x12x64xf32>) outs(%739 : tensor<1x12x?x64xf32>) permutation = [0, 2, 1, 3] 
    cf.assert %62, "must have at most one inferred (negative) dimension"
    %reshape_364 = tensor.reshape %extracted_slice_357(%from_elements_360) : (tensor<1x?x768xf32>, tensor<4xi64>) -> tensor<1x?x12x64xf32>
    %transposed_365 = linalg.transpose ins(%reshape_364 : tensor<1x?x12x64xf32>) outs(%739 : tensor<1x12x?x64xf32>) permutation = [0, 2, 1, 3] 
    %740 = linalg.generic {indexing_maps = [#map7, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%transposed_365 : tensor<1x12x?x64xf32>) outs(%739 : tensor<1x12x?x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1190 = arith.truncf %cst_148 : f64 to f32
      %1191 = arith.mulf %in, %1190 : f32
      linalg.yield %1191 : f32
    } -> tensor<1x12x?x64xf32>
    %741 = tensor.empty(%738) : tensor<1x12x64x?xf32>
    %transposed_366 = linalg.transpose ins(%reshape_361 : tensor<1x?x12x64xf32>) outs(%741 : tensor<1x12x64x?xf32>) permutation = [0, 2, 3, 1] 
    %742 = linalg.generic {indexing_maps = [#map7, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%transposed_366 : tensor<1x12x64x?xf32>) outs(%741 : tensor<1x12x64x?xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1190 = arith.truncf %cst_148 : f64 to f32
      %1191 = arith.mulf %in, %1190 : f32
      linalg.yield %1191 : f32
    } -> tensor<1x12x64x?xf32>
    cf.assert %30, "unimplemented: dynamic negative broadcast sizes"
    %743 = linalg.generic {indexing_maps = [#map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%73 : tensor<1x12x?x64xf32>) {
    ^bb0(%out: f32):
      %1190 = linalg.index 1 : index
      %1191 = linalg.index 2 : index
      %1192 = linalg.index 3 : index
      %1193 = arith.cmpi eq, %738, %c1 : index
      %1194 = arith.select %1193, %c0, %1191 : index
      %extracted = tensor.extract %740[%c0, %1190, %1194, %1192] : tensor<1x12x?x64xf32>
      linalg.yield %extracted : f32
    } -> tensor<1x12x?x64xf32>
    %collapsed_367 = tensor.collapse_shape %743 [[0, 1], [2], [3]] : tensor<1x12x?x64xf32> into tensor<12x?x64xf32>
    cf.assert %30, "unimplemented: dynamic negative broadcast sizes"
    %744 = linalg.generic {indexing_maps = [#map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%75 : tensor<1x12x64x?xf32>) {
    ^bb0(%out: f32):
      %1190 = linalg.index 1 : index
      %1191 = linalg.index 2 : index
      %1192 = linalg.index 3 : index
      %1193 = arith.cmpi eq, %738, %c1 : index
      %1194 = arith.select %1193, %c0, %1192 : index
      %extracted = tensor.extract %742[%c0, %1190, %1191, %1194] : tensor<1x12x64x?xf32>
      linalg.yield %extracted : f32
    } -> tensor<1x12x64x?xf32>
    %collapsed_368 = tensor.collapse_shape %744 [[0, 1], [2], [3]] : tensor<1x12x64x?xf32> into tensor<12x64x?xf32>
    %745 = linalg.batch_matmul ins(%collapsed_367, %collapsed_368 : tensor<12x?x64xf32>, tensor<12x64x?xf32>) outs(%78 : tensor<12x?x?xf32>) -> tensor<12x?x?xf32>
    cf.assert %85, "must have at most one inferred (negative) dimension"
    %reshape_369 = tensor.reshape %745(%from_elements_177) : (tensor<12x?x?xf32>, tensor<4xi64>) -> tensor<1x12x?x?xf32>
    cf.assert %90, "mismatched size for broadcast"
    cf.assert %90, "mismatched size for broadcast"
    %746 = linalg.generic {indexing_maps = [#map7, #map14, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%reshape_369, %69 : tensor<1x12x?x?xf32>, tensor<1x1x?x?xf32>) outs(%91 : tensor<1x12x?x?xf32>) {
    ^bb0(%in: f32, %in_495: f32, %out: f32):
      %1190 = arith.addf %in, %in_495 : f32
      linalg.yield %1190 : f32
    } -> tensor<1x12x?x?xf32>
    %747:2 = linalg.generic {indexing_maps = [#map7, #map15, #map15], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%746 : tensor<1x12x?x?xf32>) outs(%96, %94 : tensor<1x12x?xf32>, tensor<1x12x?xi64>) {
    ^bb0(%in: f32, %out: f32, %out_495: i64):
      %1190 = linalg.index 3 : index
      %1191 = arith.index_cast %1190 : index to i64
      %1192 = arith.maximumf %in, %out : f32
      %1193 = arith.cmpf ogt, %in, %out : f32
      %1194 = arith.select %1193, %1191, %out_495 : i64
      linalg.yield %1192, %1194 : f32, i64
    } -> (tensor<1x12x?xf32>, tensor<1x12x?xi64>)
    %expanded_370 = tensor.expand_shape %747#0 [[0], [1], [2, 3]] output_shape [1, 12, %89, 1] : tensor<1x12x?xf32> into tensor<1x12x?x1xf32>
    %748 = linalg.generic {indexing_maps = [#map7, #map6, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%746, %expanded_370 : tensor<1x12x?x?xf32>, tensor<1x12x?x1xf32>) outs(%91 : tensor<1x12x?x?xf32>) {
    ^bb0(%in: f32, %in_495: f32, %out: f32):
      %1190 = arith.subf %in, %in_495 : f32
      linalg.yield %1190 : f32
    } -> tensor<1x12x?x?xf32>
    %749 = linalg.generic {indexing_maps = [#map7, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%748 : tensor<1x12x?x?xf32>) outs(%91 : tensor<1x12x?x?xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1190 = math.exp %in : f32
      linalg.yield %1190 : f32
    } -> tensor<1x12x?x?xf32>
    %750 = linalg.generic {indexing_maps = [#map7, #map6], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%749 : tensor<1x12x?x?xf32>) outs(%101 : tensor<1x12x?x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1190 = arith.addf %in, %out : f32
      linalg.yield %1190 : f32
    } -> tensor<1x12x?x1xf32>
    %751 = linalg.generic {indexing_maps = [#map7, #map6, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%749, %750 : tensor<1x12x?x?xf32>, tensor<1x12x?x1xf32>) outs(%91 : tensor<1x12x?x?xf32>) {
    ^bb0(%in: f32, %in_495: f32, %out: f32):
      %1190 = arith.divf %in, %in_495 : f32
      linalg.yield %1190 : f32
    } -> tensor<1x12x?x?xf32>
    %752 = linalg.generic {indexing_maps = [#map7, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%746 : tensor<1x12x?x?xf32>) outs(%104 : tensor<1x12x?x?xi1>) {
    ^bb0(%in: f32, %out: i1):
      %1190 = arith.extf %in : f32 to f64
      %1191 = arith.cmpf oeq, %1190, %cst_149 : f64
      linalg.yield %1191 : i1
    } -> tensor<1x12x?x?xi1>
    %753 = linalg.generic {indexing_maps = [#map7, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%752 : tensor<1x12x?x?xi1>) outs(%104 : tensor<1x12x?x?xi1>) {
    ^bb0(%in: i1, %out: i1):
      %1190 = arith.cmpi eq, %in, %false : i1
      linalg.yield %1190 : i1
    } -> tensor<1x12x?x?xi1>
    %754 = linalg.generic {indexing_maps = [#map7, #map6], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%753 : tensor<1x12x?x?xi1>) outs(%108 : tensor<1x12x?x1xi1>) {
    ^bb0(%in: i1, %out: i1):
      %1190 = arith.ori %in, %out : i1
      linalg.yield %1190 : i1
    } -> tensor<1x12x?x1xi1>
    %755 = linalg.generic {indexing_maps = [#map7, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%754 : tensor<1x12x?x1xi1>) outs(%107 : tensor<1x12x?x1xi1>) {
    ^bb0(%in: i1, %out: i1):
      %1190 = arith.cmpi eq, %in, %false : i1
      linalg.yield %1190 : i1
    } -> tensor<1x12x?x1xi1>
    cf.assert %111, "negative values not allowed in new dimensions"
    cf.assert %111, "negative values not allowed in new dimensions"
    cf.assert %115, "mismatched size for broadcast"
    cf.assert %116, "mismatched size for broadcast"
    %756 = linalg.generic {indexing_maps = [#map6, #map7, #map7, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%755, %114, %751 : tensor<1x12x?x1xi1>, tensor<1x12x?x?xf32>, tensor<1x12x?x?xf32>) outs(%117 : tensor<1x12x?x?xf32>) {
    ^bb0(%in: i1, %in_495: f32, %in_496: f32, %out: f32):
      %1190 = arith.select %in, %in_495, %in_496 : f32
      linalg.yield %1190 : f32
    } -> tensor<1x12x?x?xf32>
    cf.assert %30, "unimplemented: dynamic negative broadcast sizes"
    cf.assert %30, "unimplemented: dynamic negative broadcast sizes"
    %757 = linalg.generic {indexing_maps = [#map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%119 : tensor<1x12x?x?xf32>) {
    ^bb0(%out: f32):
      %1190 = linalg.index 1 : index
      %1191 = linalg.index 2 : index
      %1192 = linalg.index 3 : index
      %1193 = arith.cmpi eq, %89, %c1 : index
      %1194 = arith.select %1193, %c0, %1191 : index
      %1195 = arith.cmpi eq, %112, %c1 : index
      %1196 = arith.select %1195, %c0, %1192 : index
      %extracted = tensor.extract %756[%c0, %1190, %1194, %1196] : tensor<1x12x?x?xf32>
      linalg.yield %extracted : f32
    } -> tensor<1x12x?x?xf32>
    cf.assert %85, "must have at most one inferred (negative) dimension"
    %reshape_371 = tensor.reshape %757(%from_elements_180) : (tensor<1x12x?x?xf32>, tensor<3xi64>) -> tensor<12x?x?xf32>
    cf.assert %30, "unimplemented: dynamic negative broadcast sizes"
    %758 = linalg.generic {indexing_maps = [#map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%73 : tensor<1x12x?x64xf32>) {
    ^bb0(%out: f32):
      %1190 = linalg.index 1 : index
      %1191 = linalg.index 2 : index
      %1192 = linalg.index 3 : index
      %1193 = arith.cmpi eq, %738, %c1 : index
      %1194 = arith.select %1193, %c0, %1191 : index
      %extracted = tensor.extract %transposed_363[%c0, %1190, %1194, %1192] : tensor<1x12x?x64xf32>
      linalg.yield %extracted : f32
    } -> tensor<1x12x?x64xf32>
    %collapsed_372 = tensor.collapse_shape %758 [[0, 1], [2], [3]] : tensor<1x12x?x64xf32> into tensor<12x?x64xf32>
    cf.assert %122, "mismatching contracting dimension"
    %759 = linalg.batch_matmul ins(%reshape_371, %collapsed_372 : tensor<12x?x?xf32>, tensor<12x?x64xf32>) outs(%124 : tensor<12x?x64xf32>) -> tensor<12x?x64xf32>
    %expanded_373 = tensor.expand_shape %759 [[0, 1], [2], [3]] output_shape [1, 12, %89, 64] : tensor<12x?x64xf32> into tensor<1x12x?x64xf32>
    %transposed_374 = linalg.transpose ins(%expanded_373 : tensor<1x12x?x64xf32>) outs(%126 : tensor<1x?x12x64xf32>) permutation = [0, 2, 1, 3] 
    cf.assert %62, "must have at most one inferred (negative) dimension"
    %reshape_375 = tensor.reshape %transposed_374(%from_elements_185) : (tensor<1x?x12x64xf32>, tensor<3xi64>) -> tensor<1x?x768xf32>
    %collapsed_376 = tensor.collapse_shape %reshape_375 [[0, 1], [2]] : tensor<1x?x768xf32> into tensor<?x768xf32>
    %760 = arith.index_cast %129 : i64 to index
    %761 = tensor.empty(%760) : tensor<?x768xf32>
    %762 = linalg.fill ins(%cst_0 : f32) outs(%761 : tensor<?x768xf32>) -> tensor<?x768xf32>
    %763 = linalg.matmul ins(%collapsed_376, %cst_61 : tensor<?x768xf32>, tensor<768x768xf32>) outs(%762 : tensor<?x768xf32>) -> tensor<?x768xf32>
    %764 = linalg.generic {indexing_maps = [#map3, #map13, #map3], iterator_types = ["parallel", "parallel"]} ins(%763, %cst_60 : tensor<?x768xf32>, tensor<768xf32>) outs(%761 : tensor<?x768xf32>) {
    ^bb0(%in: f32, %in_495: f32, %out: f32):
      %1190 = arith.addf %in, %in_495 : f32
      linalg.yield %1190 : f32
    } -> tensor<?x768xf32>
    %expanded_377 = tensor.expand_shape %764 [[0, 1], [2]] output_shape [1, %760, 768] : tensor<?x768xf32> into tensor<1x?x768xf32>
    %765 = arith.cmpi eq, %760, %670 : index
    cf.assert %765, "mismatched size for broadcast"
    %766 = tensor.empty(%760) : tensor<1x?x768xf32>
    %767 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%expanded_377, %714 : tensor<1x?x768xf32>, tensor<1x?x768xf32>) outs(%766 : tensor<1x?x768xf32>) {
    ^bb0(%in: f32, %in_495: f32, %out: f32):
      %1190 = arith.addf %in, %in_495 : f32
      linalg.yield %1190 : f32
    } -> tensor<1x?x768xf32>
    %768 = tensor.empty(%760) : tensor<1x?x768xf64>
    %769 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%767 : tensor<1x?x768xf32>) outs(%768 : tensor<1x?x768xf64>) {
    ^bb0(%in: f32, %out: f64):
      %1190 = arith.extf %in : f32 to f64
      linalg.yield %1190 : f64
    } -> tensor<1x?x768xf64>
    %770 = tensor.empty(%760) : tensor<1x?x1xf64>
    %771 = linalg.fill ins(%cst : f64) outs(%770 : tensor<1x?x1xf64>) -> tensor<1x?x1xf64>
    %772 = linalg.generic {indexing_maps = [#map1, #map11], iterator_types = ["parallel", "parallel", "reduction"]} ins(%769 : tensor<1x?x768xf64>) outs(%771 : tensor<1x?x1xf64>) {
    ^bb0(%in: f64, %out: f64):
      %1190 = arith.addf %in, %out : f64
      linalg.yield %1190 : f64
    } -> tensor<1x?x1xf64>
    %773 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%772 : tensor<1x?x1xf64>) outs(%770 : tensor<1x?x1xf64>) {
    ^bb0(%in: f64, %out: f64):
      %1190 = arith.divf %in, %cst_157 : f64
      linalg.yield %1190 : f64
    } -> tensor<1x?x1xf64>
    %774 = linalg.generic {indexing_maps = [#map1, #map11, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%769, %773 : tensor<1x?x768xf64>, tensor<1x?x1xf64>) outs(%768 : tensor<1x?x768xf64>) {
    ^bb0(%in: f64, %in_495: f64, %out: f64):
      %1190 = arith.subf %in, %in_495 : f64
      linalg.yield %1190 : f64
    } -> tensor<1x?x768xf64>
    %775 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%774, %774 : tensor<1x?x768xf64>, tensor<1x?x768xf64>) outs(%768 : tensor<1x?x768xf64>) {
    ^bb0(%in: f64, %in_495: f64, %out: f64):
      %1190 = arith.mulf %in, %in_495 : f64
      linalg.yield %1190 : f64
    } -> tensor<1x?x768xf64>
    %776 = linalg.generic {indexing_maps = [#map1, #map11], iterator_types = ["parallel", "parallel", "reduction"]} ins(%775 : tensor<1x?x768xf64>) outs(%771 : tensor<1x?x1xf64>) {
    ^bb0(%in: f64, %out: f64):
      %1190 = arith.addf %in, %out : f64
      linalg.yield %1190 : f64
    } -> tensor<1x?x1xf64>
    %777 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%776 : tensor<1x?x1xf64>) outs(%770 : tensor<1x?x1xf64>) {
    ^bb0(%in: f64, %out: f64):
      %1190 = arith.divf %in, %cst_157 : f64
      linalg.yield %1190 : f64
    } -> tensor<1x?x1xf64>
    %778 = tensor.empty(%760) : tensor<1x?x1xf32>
    %779 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%777 : tensor<1x?x1xf64>) outs(%778 : tensor<1x?x1xf32>) {
    ^bb0(%in: f64, %out: f32):
      %1190 = arith.truncf %in : f64 to f32
      linalg.yield %1190 : f32
    } -> tensor<1x?x1xf32>
    %780 = linalg.fill ins(%cst_0 : f32) outs(%778 : tensor<1x?x1xf32>) -> tensor<1x?x1xf32>
    %781 = linalg.generic {indexing_maps = [#map1, #map11], iterator_types = ["parallel", "parallel", "reduction"]} ins(%767 : tensor<1x?x768xf32>) outs(%780 : tensor<1x?x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1190 = arith.addf %in, %out : f32
      linalg.yield %1190 : f32
    } -> tensor<1x?x1xf32>
    %782 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%781 : tensor<1x?x1xf32>) outs(%778 : tensor<1x?x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1190 = arith.divf %in, %cst_158 : f32
      linalg.yield %1190 : f32
    } -> tensor<1x?x1xf32>
    %783 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%779 : tensor<1x?x1xf32>) outs(%778 : tensor<1x?x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1190 = arith.truncf %cst_154 : f64 to f32
      %1191 = arith.addf %in, %1190 : f32
      linalg.yield %1191 : f32
    } -> tensor<1x?x1xf32>
    %784 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%783 : tensor<1x?x1xf32>) outs(%778 : tensor<1x?x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1190 = math.rsqrt %in : f32
      linalg.yield %1190 : f32
    } -> tensor<1x?x1xf32>
    %785 = linalg.generic {indexing_maps = [#map1, #map11, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%767, %782 : tensor<1x?x768xf32>, tensor<1x?x1xf32>) outs(%766 : tensor<1x?x768xf32>) {
    ^bb0(%in: f32, %in_495: f32, %out: f32):
      %1190 = arith.subf %in, %in_495 : f32
      linalg.yield %1190 : f32
    } -> tensor<1x?x768xf32>
    %786 = linalg.generic {indexing_maps = [#map1, #map11, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%785, %784 : tensor<1x?x768xf32>, tensor<1x?x1xf32>) outs(%766 : tensor<1x?x768xf32>) {
    ^bb0(%in: f32, %in_495: f32, %out: f32):
      %1190 = arith.mulf %in, %in_495 : f32
      linalg.yield %1190 : f32
    } -> tensor<1x?x768xf32>
    %787 = linalg.generic {indexing_maps = [#map1, #map12, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%786, %cst_59 : tensor<1x?x768xf32>, tensor<768xf32>) outs(%766 : tensor<1x?x768xf32>) {
    ^bb0(%in: f32, %in_495: f32, %out: f32):
      %1190 = arith.mulf %in, %in_495 : f32
      linalg.yield %1190 : f32
    } -> tensor<1x?x768xf32>
    %788 = linalg.generic {indexing_maps = [#map1, #map12, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%787, %cst_58 : tensor<1x?x768xf32>, tensor<768xf32>) outs(%766 : tensor<1x?x768xf32>) {
    ^bb0(%in: f32, %in_495: f32, %out: f32):
      %1190 = arith.addf %in, %in_495 : f32
      linalg.yield %1190 : f32
    } -> tensor<1x?x768xf32>
    %collapsed_378 = tensor.collapse_shape %788 [[0, 1], [2]] : tensor<1x?x768xf32> into tensor<?x768xf32>
    %789 = tensor.empty(%760) : tensor<?x3072xf32>
    %790 = linalg.fill ins(%cst_0 : f32) outs(%789 : tensor<?x3072xf32>) -> tensor<?x3072xf32>
    %791 = linalg.matmul ins(%collapsed_378, %cst_57 : tensor<?x768xf32>, tensor<768x3072xf32>) outs(%790 : tensor<?x3072xf32>) -> tensor<?x3072xf32>
    %792 = linalg.generic {indexing_maps = [#map3, #map13, #map3], iterator_types = ["parallel", "parallel"]} ins(%791, %cst_56 : tensor<?x3072xf32>, tensor<3072xf32>) outs(%789 : tensor<?x3072xf32>) {
    ^bb0(%in: f32, %in_495: f32, %out: f32):
      %1190 = arith.addf %in, %in_495 : f32
      linalg.yield %1190 : f32
    } -> tensor<?x3072xf32>
    %expanded_379 = tensor.expand_shape %792 [[0, 1], [2]] output_shape [1, %760, 3072] : tensor<?x3072xf32> into tensor<1x?x3072xf32>
    %793 = tensor.empty(%760) : tensor<1x?x3072xf32>
    %794 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%expanded_379 : tensor<1x?x3072xf32>) outs(%793 : tensor<1x?x3072xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1190 = arith.mulf %in, %cst_159 : f32
      linalg.yield %1190 : f32
    } -> tensor<1x?x3072xf32>
    %795 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%expanded_379 : tensor<1x?x3072xf32>) outs(%793 : tensor<1x?x3072xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1190 = math.fpowi %in, %c3_i64 : f32, i64
      linalg.yield %1190 : f32
    } -> tensor<1x?x3072xf32>
    %796 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%795 : tensor<1x?x3072xf32>) outs(%793 : tensor<1x?x3072xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1190 = arith.truncf %cst_141 : f64 to f32
      %1191 = arith.mulf %in, %1190 : f32
      linalg.yield %1191 : f32
    } -> tensor<1x?x3072xf32>
    %797 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%expanded_379, %796 : tensor<1x?x3072xf32>, tensor<1x?x3072xf32>) outs(%793 : tensor<1x?x3072xf32>) {
    ^bb0(%in: f32, %in_495: f32, %out: f32):
      %1190 = arith.addf %in, %in_495 : f32
      linalg.yield %1190 : f32
    } -> tensor<1x?x3072xf32>
    %798 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%797 : tensor<1x?x3072xf32>) outs(%793 : tensor<1x?x3072xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1190 = arith.truncf %cst_140 : f64 to f32
      %1191 = arith.mulf %in, %1190 : f32
      linalg.yield %1191 : f32
    } -> tensor<1x?x3072xf32>
    %799 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%798 : tensor<1x?x3072xf32>) outs(%793 : tensor<1x?x3072xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1190 = math.tanh %in : f32
      linalg.yield %1190 : f32
    } -> tensor<1x?x3072xf32>
    %800 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%799 : tensor<1x?x3072xf32>) outs(%793 : tensor<1x?x3072xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1190 = arith.addf %in, %cst_160 : f32
      linalg.yield %1190 : f32
    } -> tensor<1x?x3072xf32>
    %801 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%794, %800 : tensor<1x?x3072xf32>, tensor<1x?x3072xf32>) outs(%793 : tensor<1x?x3072xf32>) {
    ^bb0(%in: f32, %in_495: f32, %out: f32):
      %1190 = arith.mulf %in, %in_495 : f32
      linalg.yield %1190 : f32
    } -> tensor<1x?x3072xf32>
    %collapsed_380 = tensor.collapse_shape %801 [[0, 1], [2]] : tensor<1x?x3072xf32> into tensor<?x3072xf32>
    %802 = linalg.matmul ins(%collapsed_380, %cst_55 : tensor<?x3072xf32>, tensor<3072x768xf32>) outs(%762 : tensor<?x768xf32>) -> tensor<?x768xf32>
    %803 = linalg.generic {indexing_maps = [#map3, #map13, #map3], iterator_types = ["parallel", "parallel"]} ins(%802, %cst_54 : tensor<?x768xf32>, tensor<768xf32>) outs(%761 : tensor<?x768xf32>) {
    ^bb0(%in: f32, %in_495: f32, %out: f32):
      %1190 = arith.addf %in, %in_495 : f32
      linalg.yield %1190 : f32
    } -> tensor<?x768xf32>
    %expanded_381 = tensor.expand_shape %803 [[0, 1], [2]] output_shape [1, %760, 768] : tensor<?x768xf32> into tensor<1x?x768xf32>
    %804 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%767, %expanded_381 : tensor<1x?x768xf32>, tensor<1x?x768xf32>) outs(%766 : tensor<1x?x768xf32>) {
    ^bb0(%in: f32, %in_495: f32, %out: f32):
      %1190 = arith.addf %in, %in_495 : f32
      linalg.yield %1190 : f32
    } -> tensor<1x?x768xf32>
    %805 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%804 : tensor<1x?x768xf32>) outs(%768 : tensor<1x?x768xf64>) {
    ^bb0(%in: f32, %out: f64):
      %1190 = arith.extf %in : f32 to f64
      linalg.yield %1190 : f64
    } -> tensor<1x?x768xf64>
    %806 = linalg.generic {indexing_maps = [#map1, #map11], iterator_types = ["parallel", "parallel", "reduction"]} ins(%805 : tensor<1x?x768xf64>) outs(%771 : tensor<1x?x1xf64>) {
    ^bb0(%in: f64, %out: f64):
      %1190 = arith.addf %in, %out : f64
      linalg.yield %1190 : f64
    } -> tensor<1x?x1xf64>
    %807 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%806 : tensor<1x?x1xf64>) outs(%770 : tensor<1x?x1xf64>) {
    ^bb0(%in: f64, %out: f64):
      %1190 = arith.divf %in, %cst_157 : f64
      linalg.yield %1190 : f64
    } -> tensor<1x?x1xf64>
    %808 = linalg.generic {indexing_maps = [#map1, #map11, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%805, %807 : tensor<1x?x768xf64>, tensor<1x?x1xf64>) outs(%768 : tensor<1x?x768xf64>) {
    ^bb0(%in: f64, %in_495: f64, %out: f64):
      %1190 = arith.subf %in, %in_495 : f64
      linalg.yield %1190 : f64
    } -> tensor<1x?x768xf64>
    %809 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%808, %808 : tensor<1x?x768xf64>, tensor<1x?x768xf64>) outs(%768 : tensor<1x?x768xf64>) {
    ^bb0(%in: f64, %in_495: f64, %out: f64):
      %1190 = arith.mulf %in, %in_495 : f64
      linalg.yield %1190 : f64
    } -> tensor<1x?x768xf64>
    %810 = linalg.generic {indexing_maps = [#map1, #map11], iterator_types = ["parallel", "parallel", "reduction"]} ins(%809 : tensor<1x?x768xf64>) outs(%771 : tensor<1x?x1xf64>) {
    ^bb0(%in: f64, %out: f64):
      %1190 = arith.addf %in, %out : f64
      linalg.yield %1190 : f64
    } -> tensor<1x?x1xf64>
    %811 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%810 : tensor<1x?x1xf64>) outs(%770 : tensor<1x?x1xf64>) {
    ^bb0(%in: f64, %out: f64):
      %1190 = arith.divf %in, %cst_157 : f64
      linalg.yield %1190 : f64
    } -> tensor<1x?x1xf64>
    %812 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%811 : tensor<1x?x1xf64>) outs(%778 : tensor<1x?x1xf32>) {
    ^bb0(%in: f64, %out: f32):
      %1190 = arith.truncf %in : f64 to f32
      linalg.yield %1190 : f32
    } -> tensor<1x?x1xf32>
    %813 = linalg.generic {indexing_maps = [#map1, #map11], iterator_types = ["parallel", "parallel", "reduction"]} ins(%804 : tensor<1x?x768xf32>) outs(%780 : tensor<1x?x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1190 = arith.addf %in, %out : f32
      linalg.yield %1190 : f32
    } -> tensor<1x?x1xf32>
    %814 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%813 : tensor<1x?x1xf32>) outs(%778 : tensor<1x?x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1190 = arith.divf %in, %cst_158 : f32
      linalg.yield %1190 : f32
    } -> tensor<1x?x1xf32>
    %815 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%812 : tensor<1x?x1xf32>) outs(%778 : tensor<1x?x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1190 = arith.truncf %cst_154 : f64 to f32
      %1191 = arith.addf %in, %1190 : f32
      linalg.yield %1191 : f32
    } -> tensor<1x?x1xf32>
    %816 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%815 : tensor<1x?x1xf32>) outs(%778 : tensor<1x?x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1190 = math.rsqrt %in : f32
      linalg.yield %1190 : f32
    } -> tensor<1x?x1xf32>
    %817 = linalg.generic {indexing_maps = [#map1, #map11, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%804, %814 : tensor<1x?x768xf32>, tensor<1x?x1xf32>) outs(%766 : tensor<1x?x768xf32>) {
    ^bb0(%in: f32, %in_495: f32, %out: f32):
      %1190 = arith.subf %in, %in_495 : f32
      linalg.yield %1190 : f32
    } -> tensor<1x?x768xf32>
    %818 = linalg.generic {indexing_maps = [#map1, #map11, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%817, %816 : tensor<1x?x768xf32>, tensor<1x?x1xf32>) outs(%766 : tensor<1x?x768xf32>) {
    ^bb0(%in: f32, %in_495: f32, %out: f32):
      %1190 = arith.mulf %in, %in_495 : f32
      linalg.yield %1190 : f32
    } -> tensor<1x?x768xf32>
    %819 = linalg.generic {indexing_maps = [#map1, #map12, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%818, %cst_53 : tensor<1x?x768xf32>, tensor<768xf32>) outs(%766 : tensor<1x?x768xf32>) {
    ^bb0(%in: f32, %in_495: f32, %out: f32):
      %1190 = arith.mulf %in, %in_495 : f32
      linalg.yield %1190 : f32
    } -> tensor<1x?x768xf32>
    %820 = linalg.generic {indexing_maps = [#map1, #map12, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%819, %cst_52 : tensor<1x?x768xf32>, tensor<768xf32>) outs(%766 : tensor<1x?x768xf32>) {
    ^bb0(%in: f32, %in_495: f32, %out: f32):
      %1190 = arith.addf %in, %in_495 : f32
      linalg.yield %1190 : f32
    } -> tensor<1x?x768xf32>
    %collapsed_382 = tensor.collapse_shape %820 [[0, 1], [2]] : tensor<1x?x768xf32> into tensor<?x768xf32>
    %821 = tensor.empty(%760) : tensor<?x2304xf32>
    %822 = linalg.fill ins(%cst_0 : f32) outs(%821 : tensor<?x2304xf32>) -> tensor<?x2304xf32>
    %823 = linalg.matmul ins(%collapsed_382, %cst_51 : tensor<?x768xf32>, tensor<768x2304xf32>) outs(%822 : tensor<?x2304xf32>) -> tensor<?x2304xf32>
    %824 = linalg.generic {indexing_maps = [#map3, #map13, #map3], iterator_types = ["parallel", "parallel"]} ins(%823, %cst_50 : tensor<?x2304xf32>, tensor<2304xf32>) outs(%821 : tensor<?x2304xf32>) {
    ^bb0(%in: f32, %in_495: f32, %out: f32):
      %1190 = arith.addf %in, %in_495 : f32
      linalg.yield %1190 : f32
    } -> tensor<?x2304xf32>
    %expanded_383 = tensor.expand_shape %824 [[0, 1], [2]] output_shape [1, %760, 2304] : tensor<?x2304xf32> into tensor<1x?x2304xf32>
    %extracted_slice_384 = tensor.extract_slice %expanded_383[0, 0, 0] [1, %760, 768] [1, 1, 1] : tensor<1x?x2304xf32> to tensor<1x?x768xf32>
    %extracted_slice_385 = tensor.extract_slice %expanded_383[0, 0, 768] [1, %760, 768] [1, 1, 1] : tensor<1x?x2304xf32> to tensor<1x?x768xf32>
    %extracted_slice_386 = tensor.extract_slice %expanded_383[0, 0, 1536] [1, %760, 768] [1, 1, 1] : tensor<1x?x2304xf32> to tensor<1x?x768xf32>
    cf.assert %62, "must have at most one inferred (negative) dimension"
    %825 = arith.muli %129, %c768_i64 : i64
    %826 = arith.divsi %825, %61 : i64
    %827 = arith.select %1, %826, %0 : i64
    %from_elements_387 = tensor.from_elements %c1_i64, %827, %826, %c64_i64 : tensor<4xi64>
    %reshape_388 = tensor.reshape %extracted_slice_385(%from_elements_387) : (tensor<1x?x768xf32>, tensor<4xi64>) -> tensor<1x?x12x64xf32>
    %828 = arith.index_cast %827 : i64 to index
    cf.assert %62, "must have at most one inferred (negative) dimension"
    %reshape_389 = tensor.reshape %extracted_slice_386(%from_elements_387) : (tensor<1x?x768xf32>, tensor<4xi64>) -> tensor<1x?x12x64xf32>
    %829 = tensor.empty(%828) : tensor<1x12x?x64xf32>
    %transposed_390 = linalg.transpose ins(%reshape_389 : tensor<1x?x12x64xf32>) outs(%829 : tensor<1x12x?x64xf32>) permutation = [0, 2, 1, 3] 
    cf.assert %62, "must have at most one inferred (negative) dimension"
    %reshape_391 = tensor.reshape %extracted_slice_384(%from_elements_387) : (tensor<1x?x768xf32>, tensor<4xi64>) -> tensor<1x?x12x64xf32>
    %transposed_392 = linalg.transpose ins(%reshape_391 : tensor<1x?x12x64xf32>) outs(%829 : tensor<1x12x?x64xf32>) permutation = [0, 2, 1, 3] 
    %830 = linalg.generic {indexing_maps = [#map7, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%transposed_392 : tensor<1x12x?x64xf32>) outs(%829 : tensor<1x12x?x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1190 = arith.truncf %cst_148 : f64 to f32
      %1191 = arith.mulf %in, %1190 : f32
      linalg.yield %1191 : f32
    } -> tensor<1x12x?x64xf32>
    %831 = tensor.empty(%828) : tensor<1x12x64x?xf32>
    %transposed_393 = linalg.transpose ins(%reshape_388 : tensor<1x?x12x64xf32>) outs(%831 : tensor<1x12x64x?xf32>) permutation = [0, 2, 3, 1] 
    %832 = linalg.generic {indexing_maps = [#map7, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%transposed_393 : tensor<1x12x64x?xf32>) outs(%831 : tensor<1x12x64x?xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1190 = arith.truncf %cst_148 : f64 to f32
      %1191 = arith.mulf %in, %1190 : f32
      linalg.yield %1191 : f32
    } -> tensor<1x12x64x?xf32>
    cf.assert %30, "unimplemented: dynamic negative broadcast sizes"
    %833 = linalg.generic {indexing_maps = [#map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%73 : tensor<1x12x?x64xf32>) {
    ^bb0(%out: f32):
      %1190 = linalg.index 1 : index
      %1191 = linalg.index 2 : index
      %1192 = linalg.index 3 : index
      %1193 = arith.cmpi eq, %828, %c1 : index
      %1194 = arith.select %1193, %c0, %1191 : index
      %extracted = tensor.extract %830[%c0, %1190, %1194, %1192] : tensor<1x12x?x64xf32>
      linalg.yield %extracted : f32
    } -> tensor<1x12x?x64xf32>
    %collapsed_394 = tensor.collapse_shape %833 [[0, 1], [2], [3]] : tensor<1x12x?x64xf32> into tensor<12x?x64xf32>
    cf.assert %30, "unimplemented: dynamic negative broadcast sizes"
    %834 = linalg.generic {indexing_maps = [#map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%75 : tensor<1x12x64x?xf32>) {
    ^bb0(%out: f32):
      %1190 = linalg.index 1 : index
      %1191 = linalg.index 2 : index
      %1192 = linalg.index 3 : index
      %1193 = arith.cmpi eq, %828, %c1 : index
      %1194 = arith.select %1193, %c0, %1192 : index
      %extracted = tensor.extract %832[%c0, %1190, %1191, %1194] : tensor<1x12x64x?xf32>
      linalg.yield %extracted : f32
    } -> tensor<1x12x64x?xf32>
    %collapsed_395 = tensor.collapse_shape %834 [[0, 1], [2], [3]] : tensor<1x12x64x?xf32> into tensor<12x64x?xf32>
    %835 = linalg.batch_matmul ins(%collapsed_394, %collapsed_395 : tensor<12x?x64xf32>, tensor<12x64x?xf32>) outs(%78 : tensor<12x?x?xf32>) -> tensor<12x?x?xf32>
    cf.assert %85, "must have at most one inferred (negative) dimension"
    %reshape_396 = tensor.reshape %835(%from_elements_177) : (tensor<12x?x?xf32>, tensor<4xi64>) -> tensor<1x12x?x?xf32>
    cf.assert %90, "mismatched size for broadcast"
    cf.assert %90, "mismatched size for broadcast"
    %836 = linalg.generic {indexing_maps = [#map7, #map14, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%reshape_396, %69 : tensor<1x12x?x?xf32>, tensor<1x1x?x?xf32>) outs(%91 : tensor<1x12x?x?xf32>) {
    ^bb0(%in: f32, %in_495: f32, %out: f32):
      %1190 = arith.addf %in, %in_495 : f32
      linalg.yield %1190 : f32
    } -> tensor<1x12x?x?xf32>
    %837:2 = linalg.generic {indexing_maps = [#map7, #map15, #map15], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%836 : tensor<1x12x?x?xf32>) outs(%96, %94 : tensor<1x12x?xf32>, tensor<1x12x?xi64>) {
    ^bb0(%in: f32, %out: f32, %out_495: i64):
      %1190 = linalg.index 3 : index
      %1191 = arith.index_cast %1190 : index to i64
      %1192 = arith.maximumf %in, %out : f32
      %1193 = arith.cmpf ogt, %in, %out : f32
      %1194 = arith.select %1193, %1191, %out_495 : i64
      linalg.yield %1192, %1194 : f32, i64
    } -> (tensor<1x12x?xf32>, tensor<1x12x?xi64>)
    %expanded_397 = tensor.expand_shape %837#0 [[0], [1], [2, 3]] output_shape [1, 12, %89, 1] : tensor<1x12x?xf32> into tensor<1x12x?x1xf32>
    %838 = linalg.generic {indexing_maps = [#map7, #map6, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%836, %expanded_397 : tensor<1x12x?x?xf32>, tensor<1x12x?x1xf32>) outs(%91 : tensor<1x12x?x?xf32>) {
    ^bb0(%in: f32, %in_495: f32, %out: f32):
      %1190 = arith.subf %in, %in_495 : f32
      linalg.yield %1190 : f32
    } -> tensor<1x12x?x?xf32>
    %839 = linalg.generic {indexing_maps = [#map7, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%838 : tensor<1x12x?x?xf32>) outs(%91 : tensor<1x12x?x?xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1190 = math.exp %in : f32
      linalg.yield %1190 : f32
    } -> tensor<1x12x?x?xf32>
    %840 = linalg.generic {indexing_maps = [#map7, #map6], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%839 : tensor<1x12x?x?xf32>) outs(%101 : tensor<1x12x?x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1190 = arith.addf %in, %out : f32
      linalg.yield %1190 : f32
    } -> tensor<1x12x?x1xf32>
    %841 = linalg.generic {indexing_maps = [#map7, #map6, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%839, %840 : tensor<1x12x?x?xf32>, tensor<1x12x?x1xf32>) outs(%91 : tensor<1x12x?x?xf32>) {
    ^bb0(%in: f32, %in_495: f32, %out: f32):
      %1190 = arith.divf %in, %in_495 : f32
      linalg.yield %1190 : f32
    } -> tensor<1x12x?x?xf32>
    %842 = linalg.generic {indexing_maps = [#map7, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%836 : tensor<1x12x?x?xf32>) outs(%104 : tensor<1x12x?x?xi1>) {
    ^bb0(%in: f32, %out: i1):
      %1190 = arith.extf %in : f32 to f64
      %1191 = arith.cmpf oeq, %1190, %cst_149 : f64
      linalg.yield %1191 : i1
    } -> tensor<1x12x?x?xi1>
    %843 = linalg.generic {indexing_maps = [#map7, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%842 : tensor<1x12x?x?xi1>) outs(%104 : tensor<1x12x?x?xi1>) {
    ^bb0(%in: i1, %out: i1):
      %1190 = arith.cmpi eq, %in, %false : i1
      linalg.yield %1190 : i1
    } -> tensor<1x12x?x?xi1>
    %844 = linalg.generic {indexing_maps = [#map7, #map6], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%843 : tensor<1x12x?x?xi1>) outs(%108 : tensor<1x12x?x1xi1>) {
    ^bb0(%in: i1, %out: i1):
      %1190 = arith.ori %in, %out : i1
      linalg.yield %1190 : i1
    } -> tensor<1x12x?x1xi1>
    %845 = linalg.generic {indexing_maps = [#map7, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%844 : tensor<1x12x?x1xi1>) outs(%107 : tensor<1x12x?x1xi1>) {
    ^bb0(%in: i1, %out: i1):
      %1190 = arith.cmpi eq, %in, %false : i1
      linalg.yield %1190 : i1
    } -> tensor<1x12x?x1xi1>
    cf.assert %111, "negative values not allowed in new dimensions"
    cf.assert %111, "negative values not allowed in new dimensions"
    cf.assert %115, "mismatched size for broadcast"
    cf.assert %116, "mismatched size for broadcast"
    %846 = linalg.generic {indexing_maps = [#map6, #map7, #map7, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%845, %114, %841 : tensor<1x12x?x1xi1>, tensor<1x12x?x?xf32>, tensor<1x12x?x?xf32>) outs(%117 : tensor<1x12x?x?xf32>) {
    ^bb0(%in: i1, %in_495: f32, %in_496: f32, %out: f32):
      %1190 = arith.select %in, %in_495, %in_496 : f32
      linalg.yield %1190 : f32
    } -> tensor<1x12x?x?xf32>
    cf.assert %30, "unimplemented: dynamic negative broadcast sizes"
    cf.assert %30, "unimplemented: dynamic negative broadcast sizes"
    %847 = linalg.generic {indexing_maps = [#map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%119 : tensor<1x12x?x?xf32>) {
    ^bb0(%out: f32):
      %1190 = linalg.index 1 : index
      %1191 = linalg.index 2 : index
      %1192 = linalg.index 3 : index
      %1193 = arith.cmpi eq, %89, %c1 : index
      %1194 = arith.select %1193, %c0, %1191 : index
      %1195 = arith.cmpi eq, %112, %c1 : index
      %1196 = arith.select %1195, %c0, %1192 : index
      %extracted = tensor.extract %846[%c0, %1190, %1194, %1196] : tensor<1x12x?x?xf32>
      linalg.yield %extracted : f32
    } -> tensor<1x12x?x?xf32>
    cf.assert %85, "must have at most one inferred (negative) dimension"
    %reshape_398 = tensor.reshape %847(%from_elements_180) : (tensor<1x12x?x?xf32>, tensor<3xi64>) -> tensor<12x?x?xf32>
    cf.assert %30, "unimplemented: dynamic negative broadcast sizes"
    %848 = linalg.generic {indexing_maps = [#map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%73 : tensor<1x12x?x64xf32>) {
    ^bb0(%out: f32):
      %1190 = linalg.index 1 : index
      %1191 = linalg.index 2 : index
      %1192 = linalg.index 3 : index
      %1193 = arith.cmpi eq, %828, %c1 : index
      %1194 = arith.select %1193, %c0, %1191 : index
      %extracted = tensor.extract %transposed_390[%c0, %1190, %1194, %1192] : tensor<1x12x?x64xf32>
      linalg.yield %extracted : f32
    } -> tensor<1x12x?x64xf32>
    %collapsed_399 = tensor.collapse_shape %848 [[0, 1], [2], [3]] : tensor<1x12x?x64xf32> into tensor<12x?x64xf32>
    cf.assert %122, "mismatching contracting dimension"
    %849 = linalg.batch_matmul ins(%reshape_398, %collapsed_399 : tensor<12x?x?xf32>, tensor<12x?x64xf32>) outs(%124 : tensor<12x?x64xf32>) -> tensor<12x?x64xf32>
    %expanded_400 = tensor.expand_shape %849 [[0, 1], [2], [3]] output_shape [1, 12, %89, 64] : tensor<12x?x64xf32> into tensor<1x12x?x64xf32>
    %transposed_401 = linalg.transpose ins(%expanded_400 : tensor<1x12x?x64xf32>) outs(%126 : tensor<1x?x12x64xf32>) permutation = [0, 2, 1, 3] 
    cf.assert %62, "must have at most one inferred (negative) dimension"
    %reshape_402 = tensor.reshape %transposed_401(%from_elements_185) : (tensor<1x?x12x64xf32>, tensor<3xi64>) -> tensor<1x?x768xf32>
    %collapsed_403 = tensor.collapse_shape %reshape_402 [[0, 1], [2]] : tensor<1x?x768xf32> into tensor<?x768xf32>
    %850 = arith.index_cast %129 : i64 to index
    %851 = tensor.empty(%850) : tensor<?x768xf32>
    %852 = linalg.fill ins(%cst_0 : f32) outs(%851 : tensor<?x768xf32>) -> tensor<?x768xf32>
    %853 = linalg.matmul ins(%collapsed_403, %cst_49 : tensor<?x768xf32>, tensor<768x768xf32>) outs(%852 : tensor<?x768xf32>) -> tensor<?x768xf32>
    %854 = linalg.generic {indexing_maps = [#map3, #map13, #map3], iterator_types = ["parallel", "parallel"]} ins(%853, %cst_48 : tensor<?x768xf32>, tensor<768xf32>) outs(%851 : tensor<?x768xf32>) {
    ^bb0(%in: f32, %in_495: f32, %out: f32):
      %1190 = arith.addf %in, %in_495 : f32
      linalg.yield %1190 : f32
    } -> tensor<?x768xf32>
    %expanded_404 = tensor.expand_shape %854 [[0, 1], [2]] output_shape [1, %850, 768] : tensor<?x768xf32> into tensor<1x?x768xf32>
    %855 = arith.cmpi eq, %850, %760 : index
    cf.assert %855, "mismatched size for broadcast"
    %856 = tensor.empty(%850) : tensor<1x?x768xf32>
    %857 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%expanded_404, %804 : tensor<1x?x768xf32>, tensor<1x?x768xf32>) outs(%856 : tensor<1x?x768xf32>) {
    ^bb0(%in: f32, %in_495: f32, %out: f32):
      %1190 = arith.addf %in, %in_495 : f32
      linalg.yield %1190 : f32
    } -> tensor<1x?x768xf32>
    %858 = tensor.empty(%850) : tensor<1x?x768xf64>
    %859 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%857 : tensor<1x?x768xf32>) outs(%858 : tensor<1x?x768xf64>) {
    ^bb0(%in: f32, %out: f64):
      %1190 = arith.extf %in : f32 to f64
      linalg.yield %1190 : f64
    } -> tensor<1x?x768xf64>
    %860 = tensor.empty(%850) : tensor<1x?x1xf64>
    %861 = linalg.fill ins(%cst : f64) outs(%860 : tensor<1x?x1xf64>) -> tensor<1x?x1xf64>
    %862 = linalg.generic {indexing_maps = [#map1, #map11], iterator_types = ["parallel", "parallel", "reduction"]} ins(%859 : tensor<1x?x768xf64>) outs(%861 : tensor<1x?x1xf64>) {
    ^bb0(%in: f64, %out: f64):
      %1190 = arith.addf %in, %out : f64
      linalg.yield %1190 : f64
    } -> tensor<1x?x1xf64>
    %863 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%862 : tensor<1x?x1xf64>) outs(%860 : tensor<1x?x1xf64>) {
    ^bb0(%in: f64, %out: f64):
      %1190 = arith.divf %in, %cst_157 : f64
      linalg.yield %1190 : f64
    } -> tensor<1x?x1xf64>
    %864 = linalg.generic {indexing_maps = [#map1, #map11, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%859, %863 : tensor<1x?x768xf64>, tensor<1x?x1xf64>) outs(%858 : tensor<1x?x768xf64>) {
    ^bb0(%in: f64, %in_495: f64, %out: f64):
      %1190 = arith.subf %in, %in_495 : f64
      linalg.yield %1190 : f64
    } -> tensor<1x?x768xf64>
    %865 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%864, %864 : tensor<1x?x768xf64>, tensor<1x?x768xf64>) outs(%858 : tensor<1x?x768xf64>) {
    ^bb0(%in: f64, %in_495: f64, %out: f64):
      %1190 = arith.mulf %in, %in_495 : f64
      linalg.yield %1190 : f64
    } -> tensor<1x?x768xf64>
    %866 = linalg.generic {indexing_maps = [#map1, #map11], iterator_types = ["parallel", "parallel", "reduction"]} ins(%865 : tensor<1x?x768xf64>) outs(%861 : tensor<1x?x1xf64>) {
    ^bb0(%in: f64, %out: f64):
      %1190 = arith.addf %in, %out : f64
      linalg.yield %1190 : f64
    } -> tensor<1x?x1xf64>
    %867 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%866 : tensor<1x?x1xf64>) outs(%860 : tensor<1x?x1xf64>) {
    ^bb0(%in: f64, %out: f64):
      %1190 = arith.divf %in, %cst_157 : f64
      linalg.yield %1190 : f64
    } -> tensor<1x?x1xf64>
    %868 = tensor.empty(%850) : tensor<1x?x1xf32>
    %869 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%867 : tensor<1x?x1xf64>) outs(%868 : tensor<1x?x1xf32>) {
    ^bb0(%in: f64, %out: f32):
      %1190 = arith.truncf %in : f64 to f32
      linalg.yield %1190 : f32
    } -> tensor<1x?x1xf32>
    %870 = linalg.fill ins(%cst_0 : f32) outs(%868 : tensor<1x?x1xf32>) -> tensor<1x?x1xf32>
    %871 = linalg.generic {indexing_maps = [#map1, #map11], iterator_types = ["parallel", "parallel", "reduction"]} ins(%857 : tensor<1x?x768xf32>) outs(%870 : tensor<1x?x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1190 = arith.addf %in, %out : f32
      linalg.yield %1190 : f32
    } -> tensor<1x?x1xf32>
    %872 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%871 : tensor<1x?x1xf32>) outs(%868 : tensor<1x?x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1190 = arith.divf %in, %cst_158 : f32
      linalg.yield %1190 : f32
    } -> tensor<1x?x1xf32>
    %873 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%869 : tensor<1x?x1xf32>) outs(%868 : tensor<1x?x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1190 = arith.truncf %cst_154 : f64 to f32
      %1191 = arith.addf %in, %1190 : f32
      linalg.yield %1191 : f32
    } -> tensor<1x?x1xf32>
    %874 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%873 : tensor<1x?x1xf32>) outs(%868 : tensor<1x?x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1190 = math.rsqrt %in : f32
      linalg.yield %1190 : f32
    } -> tensor<1x?x1xf32>
    %875 = linalg.generic {indexing_maps = [#map1, #map11, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%857, %872 : tensor<1x?x768xf32>, tensor<1x?x1xf32>) outs(%856 : tensor<1x?x768xf32>) {
    ^bb0(%in: f32, %in_495: f32, %out: f32):
      %1190 = arith.subf %in, %in_495 : f32
      linalg.yield %1190 : f32
    } -> tensor<1x?x768xf32>
    %876 = linalg.generic {indexing_maps = [#map1, #map11, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%875, %874 : tensor<1x?x768xf32>, tensor<1x?x1xf32>) outs(%856 : tensor<1x?x768xf32>) {
    ^bb0(%in: f32, %in_495: f32, %out: f32):
      %1190 = arith.mulf %in, %in_495 : f32
      linalg.yield %1190 : f32
    } -> tensor<1x?x768xf32>
    %877 = linalg.generic {indexing_maps = [#map1, #map12, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%876, %cst_47 : tensor<1x?x768xf32>, tensor<768xf32>) outs(%856 : tensor<1x?x768xf32>) {
    ^bb0(%in: f32, %in_495: f32, %out: f32):
      %1190 = arith.mulf %in, %in_495 : f32
      linalg.yield %1190 : f32
    } -> tensor<1x?x768xf32>
    %878 = linalg.generic {indexing_maps = [#map1, #map12, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%877, %cst_46 : tensor<1x?x768xf32>, tensor<768xf32>) outs(%856 : tensor<1x?x768xf32>) {
    ^bb0(%in: f32, %in_495: f32, %out: f32):
      %1190 = arith.addf %in, %in_495 : f32
      linalg.yield %1190 : f32
    } -> tensor<1x?x768xf32>
    %collapsed_405 = tensor.collapse_shape %878 [[0, 1], [2]] : tensor<1x?x768xf32> into tensor<?x768xf32>
    %879 = tensor.empty(%850) : tensor<?x3072xf32>
    %880 = linalg.fill ins(%cst_0 : f32) outs(%879 : tensor<?x3072xf32>) -> tensor<?x3072xf32>
    %881 = linalg.matmul ins(%collapsed_405, %cst_45 : tensor<?x768xf32>, tensor<768x3072xf32>) outs(%880 : tensor<?x3072xf32>) -> tensor<?x3072xf32>
    %882 = linalg.generic {indexing_maps = [#map3, #map13, #map3], iterator_types = ["parallel", "parallel"]} ins(%881, %cst_44 : tensor<?x3072xf32>, tensor<3072xf32>) outs(%879 : tensor<?x3072xf32>) {
    ^bb0(%in: f32, %in_495: f32, %out: f32):
      %1190 = arith.addf %in, %in_495 : f32
      linalg.yield %1190 : f32
    } -> tensor<?x3072xf32>
    %expanded_406 = tensor.expand_shape %882 [[0, 1], [2]] output_shape [1, %850, 3072] : tensor<?x3072xf32> into tensor<1x?x3072xf32>
    %883 = tensor.empty(%850) : tensor<1x?x3072xf32>
    %884 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%expanded_406 : tensor<1x?x3072xf32>) outs(%883 : tensor<1x?x3072xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1190 = arith.mulf %in, %cst_159 : f32
      linalg.yield %1190 : f32
    } -> tensor<1x?x3072xf32>
    %885 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%expanded_406 : tensor<1x?x3072xf32>) outs(%883 : tensor<1x?x3072xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1190 = math.fpowi %in, %c3_i64 : f32, i64
      linalg.yield %1190 : f32
    } -> tensor<1x?x3072xf32>
    %886 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%885 : tensor<1x?x3072xf32>) outs(%883 : tensor<1x?x3072xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1190 = arith.truncf %cst_141 : f64 to f32
      %1191 = arith.mulf %in, %1190 : f32
      linalg.yield %1191 : f32
    } -> tensor<1x?x3072xf32>
    %887 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%expanded_406, %886 : tensor<1x?x3072xf32>, tensor<1x?x3072xf32>) outs(%883 : tensor<1x?x3072xf32>) {
    ^bb0(%in: f32, %in_495: f32, %out: f32):
      %1190 = arith.addf %in, %in_495 : f32
      linalg.yield %1190 : f32
    } -> tensor<1x?x3072xf32>
    %888 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%887 : tensor<1x?x3072xf32>) outs(%883 : tensor<1x?x3072xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1190 = arith.truncf %cst_140 : f64 to f32
      %1191 = arith.mulf %in, %1190 : f32
      linalg.yield %1191 : f32
    } -> tensor<1x?x3072xf32>
    %889 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%888 : tensor<1x?x3072xf32>) outs(%883 : tensor<1x?x3072xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1190 = math.tanh %in : f32
      linalg.yield %1190 : f32
    } -> tensor<1x?x3072xf32>
    %890 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%889 : tensor<1x?x3072xf32>) outs(%883 : tensor<1x?x3072xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1190 = arith.addf %in, %cst_160 : f32
      linalg.yield %1190 : f32
    } -> tensor<1x?x3072xf32>
    %891 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%884, %890 : tensor<1x?x3072xf32>, tensor<1x?x3072xf32>) outs(%883 : tensor<1x?x3072xf32>) {
    ^bb0(%in: f32, %in_495: f32, %out: f32):
      %1190 = arith.mulf %in, %in_495 : f32
      linalg.yield %1190 : f32
    } -> tensor<1x?x3072xf32>
    %collapsed_407 = tensor.collapse_shape %891 [[0, 1], [2]] : tensor<1x?x3072xf32> into tensor<?x3072xf32>
    %892 = linalg.matmul ins(%collapsed_407, %cst_43 : tensor<?x3072xf32>, tensor<3072x768xf32>) outs(%852 : tensor<?x768xf32>) -> tensor<?x768xf32>
    %893 = linalg.generic {indexing_maps = [#map3, #map13, #map3], iterator_types = ["parallel", "parallel"]} ins(%892, %cst_42 : tensor<?x768xf32>, tensor<768xf32>) outs(%851 : tensor<?x768xf32>) {
    ^bb0(%in: f32, %in_495: f32, %out: f32):
      %1190 = arith.addf %in, %in_495 : f32
      linalg.yield %1190 : f32
    } -> tensor<?x768xf32>
    %expanded_408 = tensor.expand_shape %893 [[0, 1], [2]] output_shape [1, %850, 768] : tensor<?x768xf32> into tensor<1x?x768xf32>
    %894 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%857, %expanded_408 : tensor<1x?x768xf32>, tensor<1x?x768xf32>) outs(%856 : tensor<1x?x768xf32>) {
    ^bb0(%in: f32, %in_495: f32, %out: f32):
      %1190 = arith.addf %in, %in_495 : f32
      linalg.yield %1190 : f32
    } -> tensor<1x?x768xf32>
    %895 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%894 : tensor<1x?x768xf32>) outs(%858 : tensor<1x?x768xf64>) {
    ^bb0(%in: f32, %out: f64):
      %1190 = arith.extf %in : f32 to f64
      linalg.yield %1190 : f64
    } -> tensor<1x?x768xf64>
    %896 = linalg.generic {indexing_maps = [#map1, #map11], iterator_types = ["parallel", "parallel", "reduction"]} ins(%895 : tensor<1x?x768xf64>) outs(%861 : tensor<1x?x1xf64>) {
    ^bb0(%in: f64, %out: f64):
      %1190 = arith.addf %in, %out : f64
      linalg.yield %1190 : f64
    } -> tensor<1x?x1xf64>
    %897 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%896 : tensor<1x?x1xf64>) outs(%860 : tensor<1x?x1xf64>) {
    ^bb0(%in: f64, %out: f64):
      %1190 = arith.divf %in, %cst_157 : f64
      linalg.yield %1190 : f64
    } -> tensor<1x?x1xf64>
    %898 = linalg.generic {indexing_maps = [#map1, #map11, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%895, %897 : tensor<1x?x768xf64>, tensor<1x?x1xf64>) outs(%858 : tensor<1x?x768xf64>) {
    ^bb0(%in: f64, %in_495: f64, %out: f64):
      %1190 = arith.subf %in, %in_495 : f64
      linalg.yield %1190 : f64
    } -> tensor<1x?x768xf64>
    %899 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%898, %898 : tensor<1x?x768xf64>, tensor<1x?x768xf64>) outs(%858 : tensor<1x?x768xf64>) {
    ^bb0(%in: f64, %in_495: f64, %out: f64):
      %1190 = arith.mulf %in, %in_495 : f64
      linalg.yield %1190 : f64
    } -> tensor<1x?x768xf64>
    %900 = linalg.generic {indexing_maps = [#map1, #map11], iterator_types = ["parallel", "parallel", "reduction"]} ins(%899 : tensor<1x?x768xf64>) outs(%861 : tensor<1x?x1xf64>) {
    ^bb0(%in: f64, %out: f64):
      %1190 = arith.addf %in, %out : f64
      linalg.yield %1190 : f64
    } -> tensor<1x?x1xf64>
    %901 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%900 : tensor<1x?x1xf64>) outs(%860 : tensor<1x?x1xf64>) {
    ^bb0(%in: f64, %out: f64):
      %1190 = arith.divf %in, %cst_157 : f64
      linalg.yield %1190 : f64
    } -> tensor<1x?x1xf64>
    %902 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%901 : tensor<1x?x1xf64>) outs(%868 : tensor<1x?x1xf32>) {
    ^bb0(%in: f64, %out: f32):
      %1190 = arith.truncf %in : f64 to f32
      linalg.yield %1190 : f32
    } -> tensor<1x?x1xf32>
    %903 = linalg.generic {indexing_maps = [#map1, #map11], iterator_types = ["parallel", "parallel", "reduction"]} ins(%894 : tensor<1x?x768xf32>) outs(%870 : tensor<1x?x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1190 = arith.addf %in, %out : f32
      linalg.yield %1190 : f32
    } -> tensor<1x?x1xf32>
    %904 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%903 : tensor<1x?x1xf32>) outs(%868 : tensor<1x?x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1190 = arith.divf %in, %cst_158 : f32
      linalg.yield %1190 : f32
    } -> tensor<1x?x1xf32>
    %905 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%902 : tensor<1x?x1xf32>) outs(%868 : tensor<1x?x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1190 = arith.truncf %cst_154 : f64 to f32
      %1191 = arith.addf %in, %1190 : f32
      linalg.yield %1191 : f32
    } -> tensor<1x?x1xf32>
    %906 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%905 : tensor<1x?x1xf32>) outs(%868 : tensor<1x?x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1190 = math.rsqrt %in : f32
      linalg.yield %1190 : f32
    } -> tensor<1x?x1xf32>
    %907 = linalg.generic {indexing_maps = [#map1, #map11, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%894, %904 : tensor<1x?x768xf32>, tensor<1x?x1xf32>) outs(%856 : tensor<1x?x768xf32>) {
    ^bb0(%in: f32, %in_495: f32, %out: f32):
      %1190 = arith.subf %in, %in_495 : f32
      linalg.yield %1190 : f32
    } -> tensor<1x?x768xf32>
    %908 = linalg.generic {indexing_maps = [#map1, #map11, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%907, %906 : tensor<1x?x768xf32>, tensor<1x?x1xf32>) outs(%856 : tensor<1x?x768xf32>) {
    ^bb0(%in: f32, %in_495: f32, %out: f32):
      %1190 = arith.mulf %in, %in_495 : f32
      linalg.yield %1190 : f32
    } -> tensor<1x?x768xf32>
    %909 = linalg.generic {indexing_maps = [#map1, #map12, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%908, %cst_41 : tensor<1x?x768xf32>, tensor<768xf32>) outs(%856 : tensor<1x?x768xf32>) {
    ^bb0(%in: f32, %in_495: f32, %out: f32):
      %1190 = arith.mulf %in, %in_495 : f32
      linalg.yield %1190 : f32
    } -> tensor<1x?x768xf32>
    %910 = linalg.generic {indexing_maps = [#map1, #map12, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%909, %cst_40 : tensor<1x?x768xf32>, tensor<768xf32>) outs(%856 : tensor<1x?x768xf32>) {
    ^bb0(%in: f32, %in_495: f32, %out: f32):
      %1190 = arith.addf %in, %in_495 : f32
      linalg.yield %1190 : f32
    } -> tensor<1x?x768xf32>
    %collapsed_409 = tensor.collapse_shape %910 [[0, 1], [2]] : tensor<1x?x768xf32> into tensor<?x768xf32>
    %911 = tensor.empty(%850) : tensor<?x2304xf32>
    %912 = linalg.fill ins(%cst_0 : f32) outs(%911 : tensor<?x2304xf32>) -> tensor<?x2304xf32>
    %913 = linalg.matmul ins(%collapsed_409, %cst_39 : tensor<?x768xf32>, tensor<768x2304xf32>) outs(%912 : tensor<?x2304xf32>) -> tensor<?x2304xf32>
    %914 = linalg.generic {indexing_maps = [#map3, #map13, #map3], iterator_types = ["parallel", "parallel"]} ins(%913, %cst_38 : tensor<?x2304xf32>, tensor<2304xf32>) outs(%911 : tensor<?x2304xf32>) {
    ^bb0(%in: f32, %in_495: f32, %out: f32):
      %1190 = arith.addf %in, %in_495 : f32
      linalg.yield %1190 : f32
    } -> tensor<?x2304xf32>
    %expanded_410 = tensor.expand_shape %914 [[0, 1], [2]] output_shape [1, %850, 2304] : tensor<?x2304xf32> into tensor<1x?x2304xf32>
    %extracted_slice_411 = tensor.extract_slice %expanded_410[0, 0, 0] [1, %850, 768] [1, 1, 1] : tensor<1x?x2304xf32> to tensor<1x?x768xf32>
    %extracted_slice_412 = tensor.extract_slice %expanded_410[0, 0, 768] [1, %850, 768] [1, 1, 1] : tensor<1x?x2304xf32> to tensor<1x?x768xf32>
    %extracted_slice_413 = tensor.extract_slice %expanded_410[0, 0, 1536] [1, %850, 768] [1, 1, 1] : tensor<1x?x2304xf32> to tensor<1x?x768xf32>
    cf.assert %62, "must have at most one inferred (negative) dimension"
    %915 = arith.muli %129, %c768_i64 : i64
    %916 = arith.divsi %915, %61 : i64
    %917 = arith.select %1, %916, %0 : i64
    %from_elements_414 = tensor.from_elements %c1_i64, %917, %916, %c64_i64 : tensor<4xi64>
    %reshape_415 = tensor.reshape %extracted_slice_412(%from_elements_414) : (tensor<1x?x768xf32>, tensor<4xi64>) -> tensor<1x?x12x64xf32>
    %918 = arith.index_cast %917 : i64 to index
    cf.assert %62, "must have at most one inferred (negative) dimension"
    %reshape_416 = tensor.reshape %extracted_slice_413(%from_elements_414) : (tensor<1x?x768xf32>, tensor<4xi64>) -> tensor<1x?x12x64xf32>
    %919 = tensor.empty(%918) : tensor<1x12x?x64xf32>
    %transposed_417 = linalg.transpose ins(%reshape_416 : tensor<1x?x12x64xf32>) outs(%919 : tensor<1x12x?x64xf32>) permutation = [0, 2, 1, 3] 
    cf.assert %62, "must have at most one inferred (negative) dimension"
    %reshape_418 = tensor.reshape %extracted_slice_411(%from_elements_414) : (tensor<1x?x768xf32>, tensor<4xi64>) -> tensor<1x?x12x64xf32>
    %transposed_419 = linalg.transpose ins(%reshape_418 : tensor<1x?x12x64xf32>) outs(%919 : tensor<1x12x?x64xf32>) permutation = [0, 2, 1, 3] 
    %920 = linalg.generic {indexing_maps = [#map7, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%transposed_419 : tensor<1x12x?x64xf32>) outs(%919 : tensor<1x12x?x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1190 = arith.truncf %cst_148 : f64 to f32
      %1191 = arith.mulf %in, %1190 : f32
      linalg.yield %1191 : f32
    } -> tensor<1x12x?x64xf32>
    %921 = tensor.empty(%918) : tensor<1x12x64x?xf32>
    %transposed_420 = linalg.transpose ins(%reshape_415 : tensor<1x?x12x64xf32>) outs(%921 : tensor<1x12x64x?xf32>) permutation = [0, 2, 3, 1] 
    %922 = linalg.generic {indexing_maps = [#map7, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%transposed_420 : tensor<1x12x64x?xf32>) outs(%921 : tensor<1x12x64x?xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1190 = arith.truncf %cst_148 : f64 to f32
      %1191 = arith.mulf %in, %1190 : f32
      linalg.yield %1191 : f32
    } -> tensor<1x12x64x?xf32>
    cf.assert %30, "unimplemented: dynamic negative broadcast sizes"
    %923 = linalg.generic {indexing_maps = [#map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%73 : tensor<1x12x?x64xf32>) {
    ^bb0(%out: f32):
      %1190 = linalg.index 1 : index
      %1191 = linalg.index 2 : index
      %1192 = linalg.index 3 : index
      %1193 = arith.cmpi eq, %918, %c1 : index
      %1194 = arith.select %1193, %c0, %1191 : index
      %extracted = tensor.extract %920[%c0, %1190, %1194, %1192] : tensor<1x12x?x64xf32>
      linalg.yield %extracted : f32
    } -> tensor<1x12x?x64xf32>
    %collapsed_421 = tensor.collapse_shape %923 [[0, 1], [2], [3]] : tensor<1x12x?x64xf32> into tensor<12x?x64xf32>
    cf.assert %30, "unimplemented: dynamic negative broadcast sizes"
    %924 = linalg.generic {indexing_maps = [#map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%75 : tensor<1x12x64x?xf32>) {
    ^bb0(%out: f32):
      %1190 = linalg.index 1 : index
      %1191 = linalg.index 2 : index
      %1192 = linalg.index 3 : index
      %1193 = arith.cmpi eq, %918, %c1 : index
      %1194 = arith.select %1193, %c0, %1192 : index
      %extracted = tensor.extract %922[%c0, %1190, %1191, %1194] : tensor<1x12x64x?xf32>
      linalg.yield %extracted : f32
    } -> tensor<1x12x64x?xf32>
    %collapsed_422 = tensor.collapse_shape %924 [[0, 1], [2], [3]] : tensor<1x12x64x?xf32> into tensor<12x64x?xf32>
    %925 = linalg.batch_matmul ins(%collapsed_421, %collapsed_422 : tensor<12x?x64xf32>, tensor<12x64x?xf32>) outs(%78 : tensor<12x?x?xf32>) -> tensor<12x?x?xf32>
    cf.assert %85, "must have at most one inferred (negative) dimension"
    %reshape_423 = tensor.reshape %925(%from_elements_177) : (tensor<12x?x?xf32>, tensor<4xi64>) -> tensor<1x12x?x?xf32>
    cf.assert %90, "mismatched size for broadcast"
    cf.assert %90, "mismatched size for broadcast"
    %926 = linalg.generic {indexing_maps = [#map7, #map14, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%reshape_423, %69 : tensor<1x12x?x?xf32>, tensor<1x1x?x?xf32>) outs(%91 : tensor<1x12x?x?xf32>) {
    ^bb0(%in: f32, %in_495: f32, %out: f32):
      %1190 = arith.addf %in, %in_495 : f32
      linalg.yield %1190 : f32
    } -> tensor<1x12x?x?xf32>
    %927:2 = linalg.generic {indexing_maps = [#map7, #map15, #map15], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%926 : tensor<1x12x?x?xf32>) outs(%96, %94 : tensor<1x12x?xf32>, tensor<1x12x?xi64>) {
    ^bb0(%in: f32, %out: f32, %out_495: i64):
      %1190 = linalg.index 3 : index
      %1191 = arith.index_cast %1190 : index to i64
      %1192 = arith.maximumf %in, %out : f32
      %1193 = arith.cmpf ogt, %in, %out : f32
      %1194 = arith.select %1193, %1191, %out_495 : i64
      linalg.yield %1192, %1194 : f32, i64
    } -> (tensor<1x12x?xf32>, tensor<1x12x?xi64>)
    %expanded_424 = tensor.expand_shape %927#0 [[0], [1], [2, 3]] output_shape [1, 12, %89, 1] : tensor<1x12x?xf32> into tensor<1x12x?x1xf32>
    %928 = linalg.generic {indexing_maps = [#map7, #map6, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%926, %expanded_424 : tensor<1x12x?x?xf32>, tensor<1x12x?x1xf32>) outs(%91 : tensor<1x12x?x?xf32>) {
    ^bb0(%in: f32, %in_495: f32, %out: f32):
      %1190 = arith.subf %in, %in_495 : f32
      linalg.yield %1190 : f32
    } -> tensor<1x12x?x?xf32>
    %929 = linalg.generic {indexing_maps = [#map7, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%928 : tensor<1x12x?x?xf32>) outs(%91 : tensor<1x12x?x?xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1190 = math.exp %in : f32
      linalg.yield %1190 : f32
    } -> tensor<1x12x?x?xf32>
    %930 = linalg.generic {indexing_maps = [#map7, #map6], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%929 : tensor<1x12x?x?xf32>) outs(%101 : tensor<1x12x?x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1190 = arith.addf %in, %out : f32
      linalg.yield %1190 : f32
    } -> tensor<1x12x?x1xf32>
    %931 = linalg.generic {indexing_maps = [#map7, #map6, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%929, %930 : tensor<1x12x?x?xf32>, tensor<1x12x?x1xf32>) outs(%91 : tensor<1x12x?x?xf32>) {
    ^bb0(%in: f32, %in_495: f32, %out: f32):
      %1190 = arith.divf %in, %in_495 : f32
      linalg.yield %1190 : f32
    } -> tensor<1x12x?x?xf32>
    %932 = linalg.generic {indexing_maps = [#map7, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%926 : tensor<1x12x?x?xf32>) outs(%104 : tensor<1x12x?x?xi1>) {
    ^bb0(%in: f32, %out: i1):
      %1190 = arith.extf %in : f32 to f64
      %1191 = arith.cmpf oeq, %1190, %cst_149 : f64
      linalg.yield %1191 : i1
    } -> tensor<1x12x?x?xi1>
    %933 = linalg.generic {indexing_maps = [#map7, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%932 : tensor<1x12x?x?xi1>) outs(%104 : tensor<1x12x?x?xi1>) {
    ^bb0(%in: i1, %out: i1):
      %1190 = arith.cmpi eq, %in, %false : i1
      linalg.yield %1190 : i1
    } -> tensor<1x12x?x?xi1>
    %934 = linalg.generic {indexing_maps = [#map7, #map6], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%933 : tensor<1x12x?x?xi1>) outs(%108 : tensor<1x12x?x1xi1>) {
    ^bb0(%in: i1, %out: i1):
      %1190 = arith.ori %in, %out : i1
      linalg.yield %1190 : i1
    } -> tensor<1x12x?x1xi1>
    %935 = linalg.generic {indexing_maps = [#map7, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%934 : tensor<1x12x?x1xi1>) outs(%107 : tensor<1x12x?x1xi1>) {
    ^bb0(%in: i1, %out: i1):
      %1190 = arith.cmpi eq, %in, %false : i1
      linalg.yield %1190 : i1
    } -> tensor<1x12x?x1xi1>
    cf.assert %111, "negative values not allowed in new dimensions"
    cf.assert %111, "negative values not allowed in new dimensions"
    cf.assert %115, "mismatched size for broadcast"
    cf.assert %116, "mismatched size for broadcast"
    %936 = linalg.generic {indexing_maps = [#map6, #map7, #map7, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%935, %114, %931 : tensor<1x12x?x1xi1>, tensor<1x12x?x?xf32>, tensor<1x12x?x?xf32>) outs(%117 : tensor<1x12x?x?xf32>) {
    ^bb0(%in: i1, %in_495: f32, %in_496: f32, %out: f32):
      %1190 = arith.select %in, %in_495, %in_496 : f32
      linalg.yield %1190 : f32
    } -> tensor<1x12x?x?xf32>
    cf.assert %30, "unimplemented: dynamic negative broadcast sizes"
    cf.assert %30, "unimplemented: dynamic negative broadcast sizes"
    %937 = linalg.generic {indexing_maps = [#map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%119 : tensor<1x12x?x?xf32>) {
    ^bb0(%out: f32):
      %1190 = linalg.index 1 : index
      %1191 = linalg.index 2 : index
      %1192 = linalg.index 3 : index
      %1193 = arith.cmpi eq, %89, %c1 : index
      %1194 = arith.select %1193, %c0, %1191 : index
      %1195 = arith.cmpi eq, %112, %c1 : index
      %1196 = arith.select %1195, %c0, %1192 : index
      %extracted = tensor.extract %936[%c0, %1190, %1194, %1196] : tensor<1x12x?x?xf32>
      linalg.yield %extracted : f32
    } -> tensor<1x12x?x?xf32>
    cf.assert %85, "must have at most one inferred (negative) dimension"
    %reshape_425 = tensor.reshape %937(%from_elements_180) : (tensor<1x12x?x?xf32>, tensor<3xi64>) -> tensor<12x?x?xf32>
    cf.assert %30, "unimplemented: dynamic negative broadcast sizes"
    %938 = linalg.generic {indexing_maps = [#map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%73 : tensor<1x12x?x64xf32>) {
    ^bb0(%out: f32):
      %1190 = linalg.index 1 : index
      %1191 = linalg.index 2 : index
      %1192 = linalg.index 3 : index
      %1193 = arith.cmpi eq, %918, %c1 : index
      %1194 = arith.select %1193, %c0, %1191 : index
      %extracted = tensor.extract %transposed_417[%c0, %1190, %1194, %1192] : tensor<1x12x?x64xf32>
      linalg.yield %extracted : f32
    } -> tensor<1x12x?x64xf32>
    %collapsed_426 = tensor.collapse_shape %938 [[0, 1], [2], [3]] : tensor<1x12x?x64xf32> into tensor<12x?x64xf32>
    cf.assert %122, "mismatching contracting dimension"
    %939 = linalg.batch_matmul ins(%reshape_425, %collapsed_426 : tensor<12x?x?xf32>, tensor<12x?x64xf32>) outs(%124 : tensor<12x?x64xf32>) -> tensor<12x?x64xf32>
    %expanded_427 = tensor.expand_shape %939 [[0, 1], [2], [3]] output_shape [1, 12, %89, 64] : tensor<12x?x64xf32> into tensor<1x12x?x64xf32>
    %transposed_428 = linalg.transpose ins(%expanded_427 : tensor<1x12x?x64xf32>) outs(%126 : tensor<1x?x12x64xf32>) permutation = [0, 2, 1, 3] 
    cf.assert %62, "must have at most one inferred (negative) dimension"
    %reshape_429 = tensor.reshape %transposed_428(%from_elements_185) : (tensor<1x?x12x64xf32>, tensor<3xi64>) -> tensor<1x?x768xf32>
    %collapsed_430 = tensor.collapse_shape %reshape_429 [[0, 1], [2]] : tensor<1x?x768xf32> into tensor<?x768xf32>
    %940 = arith.index_cast %129 : i64 to index
    %941 = tensor.empty(%940) : tensor<?x768xf32>
    %942 = linalg.fill ins(%cst_0 : f32) outs(%941 : tensor<?x768xf32>) -> tensor<?x768xf32>
    %943 = linalg.matmul ins(%collapsed_430, %cst_37 : tensor<?x768xf32>, tensor<768x768xf32>) outs(%942 : tensor<?x768xf32>) -> tensor<?x768xf32>
    %944 = linalg.generic {indexing_maps = [#map3, #map13, #map3], iterator_types = ["parallel", "parallel"]} ins(%943, %cst_36 : tensor<?x768xf32>, tensor<768xf32>) outs(%941 : tensor<?x768xf32>) {
    ^bb0(%in: f32, %in_495: f32, %out: f32):
      %1190 = arith.addf %in, %in_495 : f32
      linalg.yield %1190 : f32
    } -> tensor<?x768xf32>
    %expanded_431 = tensor.expand_shape %944 [[0, 1], [2]] output_shape [1, %940, 768] : tensor<?x768xf32> into tensor<1x?x768xf32>
    %945 = arith.cmpi eq, %940, %850 : index
    cf.assert %945, "mismatched size for broadcast"
    %946 = tensor.empty(%940) : tensor<1x?x768xf32>
    %947 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%expanded_431, %894 : tensor<1x?x768xf32>, tensor<1x?x768xf32>) outs(%946 : tensor<1x?x768xf32>) {
    ^bb0(%in: f32, %in_495: f32, %out: f32):
      %1190 = arith.addf %in, %in_495 : f32
      linalg.yield %1190 : f32
    } -> tensor<1x?x768xf32>
    %948 = tensor.empty(%940) : tensor<1x?x768xf64>
    %949 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%947 : tensor<1x?x768xf32>) outs(%948 : tensor<1x?x768xf64>) {
    ^bb0(%in: f32, %out: f64):
      %1190 = arith.extf %in : f32 to f64
      linalg.yield %1190 : f64
    } -> tensor<1x?x768xf64>
    %950 = tensor.empty(%940) : tensor<1x?x1xf64>
    %951 = linalg.fill ins(%cst : f64) outs(%950 : tensor<1x?x1xf64>) -> tensor<1x?x1xf64>
    %952 = linalg.generic {indexing_maps = [#map1, #map11], iterator_types = ["parallel", "parallel", "reduction"]} ins(%949 : tensor<1x?x768xf64>) outs(%951 : tensor<1x?x1xf64>) {
    ^bb0(%in: f64, %out: f64):
      %1190 = arith.addf %in, %out : f64
      linalg.yield %1190 : f64
    } -> tensor<1x?x1xf64>
    %953 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%952 : tensor<1x?x1xf64>) outs(%950 : tensor<1x?x1xf64>) {
    ^bb0(%in: f64, %out: f64):
      %1190 = arith.divf %in, %cst_157 : f64
      linalg.yield %1190 : f64
    } -> tensor<1x?x1xf64>
    %954 = linalg.generic {indexing_maps = [#map1, #map11, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%949, %953 : tensor<1x?x768xf64>, tensor<1x?x1xf64>) outs(%948 : tensor<1x?x768xf64>) {
    ^bb0(%in: f64, %in_495: f64, %out: f64):
      %1190 = arith.subf %in, %in_495 : f64
      linalg.yield %1190 : f64
    } -> tensor<1x?x768xf64>
    %955 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%954, %954 : tensor<1x?x768xf64>, tensor<1x?x768xf64>) outs(%948 : tensor<1x?x768xf64>) {
    ^bb0(%in: f64, %in_495: f64, %out: f64):
      %1190 = arith.mulf %in, %in_495 : f64
      linalg.yield %1190 : f64
    } -> tensor<1x?x768xf64>
    %956 = linalg.generic {indexing_maps = [#map1, #map11], iterator_types = ["parallel", "parallel", "reduction"]} ins(%955 : tensor<1x?x768xf64>) outs(%951 : tensor<1x?x1xf64>) {
    ^bb0(%in: f64, %out: f64):
      %1190 = arith.addf %in, %out : f64
      linalg.yield %1190 : f64
    } -> tensor<1x?x1xf64>
    %957 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%956 : tensor<1x?x1xf64>) outs(%950 : tensor<1x?x1xf64>) {
    ^bb0(%in: f64, %out: f64):
      %1190 = arith.divf %in, %cst_157 : f64
      linalg.yield %1190 : f64
    } -> tensor<1x?x1xf64>
    %958 = tensor.empty(%940) : tensor<1x?x1xf32>
    %959 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%957 : tensor<1x?x1xf64>) outs(%958 : tensor<1x?x1xf32>) {
    ^bb0(%in: f64, %out: f32):
      %1190 = arith.truncf %in : f64 to f32
      linalg.yield %1190 : f32
    } -> tensor<1x?x1xf32>
    %960 = linalg.fill ins(%cst_0 : f32) outs(%958 : tensor<1x?x1xf32>) -> tensor<1x?x1xf32>
    %961 = linalg.generic {indexing_maps = [#map1, #map11], iterator_types = ["parallel", "parallel", "reduction"]} ins(%947 : tensor<1x?x768xf32>) outs(%960 : tensor<1x?x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1190 = arith.addf %in, %out : f32
      linalg.yield %1190 : f32
    } -> tensor<1x?x1xf32>
    %962 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%961 : tensor<1x?x1xf32>) outs(%958 : tensor<1x?x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1190 = arith.divf %in, %cst_158 : f32
      linalg.yield %1190 : f32
    } -> tensor<1x?x1xf32>
    %963 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%959 : tensor<1x?x1xf32>) outs(%958 : tensor<1x?x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1190 = arith.truncf %cst_154 : f64 to f32
      %1191 = arith.addf %in, %1190 : f32
      linalg.yield %1191 : f32
    } -> tensor<1x?x1xf32>
    %964 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%963 : tensor<1x?x1xf32>) outs(%958 : tensor<1x?x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1190 = math.rsqrt %in : f32
      linalg.yield %1190 : f32
    } -> tensor<1x?x1xf32>
    %965 = linalg.generic {indexing_maps = [#map1, #map11, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%947, %962 : tensor<1x?x768xf32>, tensor<1x?x1xf32>) outs(%946 : tensor<1x?x768xf32>) {
    ^bb0(%in: f32, %in_495: f32, %out: f32):
      %1190 = arith.subf %in, %in_495 : f32
      linalg.yield %1190 : f32
    } -> tensor<1x?x768xf32>
    %966 = linalg.generic {indexing_maps = [#map1, #map11, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%965, %964 : tensor<1x?x768xf32>, tensor<1x?x1xf32>) outs(%946 : tensor<1x?x768xf32>) {
    ^bb0(%in: f32, %in_495: f32, %out: f32):
      %1190 = arith.mulf %in, %in_495 : f32
      linalg.yield %1190 : f32
    } -> tensor<1x?x768xf32>
    %967 = linalg.generic {indexing_maps = [#map1, #map12, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%966, %cst_35 : tensor<1x?x768xf32>, tensor<768xf32>) outs(%946 : tensor<1x?x768xf32>) {
    ^bb0(%in: f32, %in_495: f32, %out: f32):
      %1190 = arith.mulf %in, %in_495 : f32
      linalg.yield %1190 : f32
    } -> tensor<1x?x768xf32>
    %968 = linalg.generic {indexing_maps = [#map1, #map12, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%967, %cst_34 : tensor<1x?x768xf32>, tensor<768xf32>) outs(%946 : tensor<1x?x768xf32>) {
    ^bb0(%in: f32, %in_495: f32, %out: f32):
      %1190 = arith.addf %in, %in_495 : f32
      linalg.yield %1190 : f32
    } -> tensor<1x?x768xf32>
    %collapsed_432 = tensor.collapse_shape %968 [[0, 1], [2]] : tensor<1x?x768xf32> into tensor<?x768xf32>
    %969 = tensor.empty(%940) : tensor<?x3072xf32>
    %970 = linalg.fill ins(%cst_0 : f32) outs(%969 : tensor<?x3072xf32>) -> tensor<?x3072xf32>
    %971 = linalg.matmul ins(%collapsed_432, %cst_33 : tensor<?x768xf32>, tensor<768x3072xf32>) outs(%970 : tensor<?x3072xf32>) -> tensor<?x3072xf32>
    %972 = linalg.generic {indexing_maps = [#map3, #map13, #map3], iterator_types = ["parallel", "parallel"]} ins(%971, %cst_32 : tensor<?x3072xf32>, tensor<3072xf32>) outs(%969 : tensor<?x3072xf32>) {
    ^bb0(%in: f32, %in_495: f32, %out: f32):
      %1190 = arith.addf %in, %in_495 : f32
      linalg.yield %1190 : f32
    } -> tensor<?x3072xf32>
    %expanded_433 = tensor.expand_shape %972 [[0, 1], [2]] output_shape [1, %940, 3072] : tensor<?x3072xf32> into tensor<1x?x3072xf32>
    %973 = tensor.empty(%940) : tensor<1x?x3072xf32>
    %974 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%expanded_433 : tensor<1x?x3072xf32>) outs(%973 : tensor<1x?x3072xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1190 = arith.mulf %in, %cst_159 : f32
      linalg.yield %1190 : f32
    } -> tensor<1x?x3072xf32>
    %975 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%expanded_433 : tensor<1x?x3072xf32>) outs(%973 : tensor<1x?x3072xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1190 = math.fpowi %in, %c3_i64 : f32, i64
      linalg.yield %1190 : f32
    } -> tensor<1x?x3072xf32>
    %976 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%975 : tensor<1x?x3072xf32>) outs(%973 : tensor<1x?x3072xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1190 = arith.truncf %cst_141 : f64 to f32
      %1191 = arith.mulf %in, %1190 : f32
      linalg.yield %1191 : f32
    } -> tensor<1x?x3072xf32>
    %977 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%expanded_433, %976 : tensor<1x?x3072xf32>, tensor<1x?x3072xf32>) outs(%973 : tensor<1x?x3072xf32>) {
    ^bb0(%in: f32, %in_495: f32, %out: f32):
      %1190 = arith.addf %in, %in_495 : f32
      linalg.yield %1190 : f32
    } -> tensor<1x?x3072xf32>
    %978 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%977 : tensor<1x?x3072xf32>) outs(%973 : tensor<1x?x3072xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1190 = arith.truncf %cst_140 : f64 to f32
      %1191 = arith.mulf %in, %1190 : f32
      linalg.yield %1191 : f32
    } -> tensor<1x?x3072xf32>
    %979 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%978 : tensor<1x?x3072xf32>) outs(%973 : tensor<1x?x3072xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1190 = math.tanh %in : f32
      linalg.yield %1190 : f32
    } -> tensor<1x?x3072xf32>
    %980 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%979 : tensor<1x?x3072xf32>) outs(%973 : tensor<1x?x3072xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1190 = arith.addf %in, %cst_160 : f32
      linalg.yield %1190 : f32
    } -> tensor<1x?x3072xf32>
    %981 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%974, %980 : tensor<1x?x3072xf32>, tensor<1x?x3072xf32>) outs(%973 : tensor<1x?x3072xf32>) {
    ^bb0(%in: f32, %in_495: f32, %out: f32):
      %1190 = arith.mulf %in, %in_495 : f32
      linalg.yield %1190 : f32
    } -> tensor<1x?x3072xf32>
    %collapsed_434 = tensor.collapse_shape %981 [[0, 1], [2]] : tensor<1x?x3072xf32> into tensor<?x3072xf32>
    %982 = linalg.matmul ins(%collapsed_434, %cst_31 : tensor<?x3072xf32>, tensor<3072x768xf32>) outs(%942 : tensor<?x768xf32>) -> tensor<?x768xf32>
    %983 = linalg.generic {indexing_maps = [#map3, #map13, #map3], iterator_types = ["parallel", "parallel"]} ins(%982, %cst_30 : tensor<?x768xf32>, tensor<768xf32>) outs(%941 : tensor<?x768xf32>) {
    ^bb0(%in: f32, %in_495: f32, %out: f32):
      %1190 = arith.addf %in, %in_495 : f32
      linalg.yield %1190 : f32
    } -> tensor<?x768xf32>
    %expanded_435 = tensor.expand_shape %983 [[0, 1], [2]] output_shape [1, %940, 768] : tensor<?x768xf32> into tensor<1x?x768xf32>
    %984 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%947, %expanded_435 : tensor<1x?x768xf32>, tensor<1x?x768xf32>) outs(%946 : tensor<1x?x768xf32>) {
    ^bb0(%in: f32, %in_495: f32, %out: f32):
      %1190 = arith.addf %in, %in_495 : f32
      linalg.yield %1190 : f32
    } -> tensor<1x?x768xf32>
    %985 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%984 : tensor<1x?x768xf32>) outs(%948 : tensor<1x?x768xf64>) {
    ^bb0(%in: f32, %out: f64):
      %1190 = arith.extf %in : f32 to f64
      linalg.yield %1190 : f64
    } -> tensor<1x?x768xf64>
    %986 = linalg.generic {indexing_maps = [#map1, #map11], iterator_types = ["parallel", "parallel", "reduction"]} ins(%985 : tensor<1x?x768xf64>) outs(%951 : tensor<1x?x1xf64>) {
    ^bb0(%in: f64, %out: f64):
      %1190 = arith.addf %in, %out : f64
      linalg.yield %1190 : f64
    } -> tensor<1x?x1xf64>
    %987 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%986 : tensor<1x?x1xf64>) outs(%950 : tensor<1x?x1xf64>) {
    ^bb0(%in: f64, %out: f64):
      %1190 = arith.divf %in, %cst_157 : f64
      linalg.yield %1190 : f64
    } -> tensor<1x?x1xf64>
    %988 = linalg.generic {indexing_maps = [#map1, #map11, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%985, %987 : tensor<1x?x768xf64>, tensor<1x?x1xf64>) outs(%948 : tensor<1x?x768xf64>) {
    ^bb0(%in: f64, %in_495: f64, %out: f64):
      %1190 = arith.subf %in, %in_495 : f64
      linalg.yield %1190 : f64
    } -> tensor<1x?x768xf64>
    %989 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%988, %988 : tensor<1x?x768xf64>, tensor<1x?x768xf64>) outs(%948 : tensor<1x?x768xf64>) {
    ^bb0(%in: f64, %in_495: f64, %out: f64):
      %1190 = arith.mulf %in, %in_495 : f64
      linalg.yield %1190 : f64
    } -> tensor<1x?x768xf64>
    %990 = linalg.generic {indexing_maps = [#map1, #map11], iterator_types = ["parallel", "parallel", "reduction"]} ins(%989 : tensor<1x?x768xf64>) outs(%951 : tensor<1x?x1xf64>) {
    ^bb0(%in: f64, %out: f64):
      %1190 = arith.addf %in, %out : f64
      linalg.yield %1190 : f64
    } -> tensor<1x?x1xf64>
    %991 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%990 : tensor<1x?x1xf64>) outs(%950 : tensor<1x?x1xf64>) {
    ^bb0(%in: f64, %out: f64):
      %1190 = arith.divf %in, %cst_157 : f64
      linalg.yield %1190 : f64
    } -> tensor<1x?x1xf64>
    %992 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%991 : tensor<1x?x1xf64>) outs(%958 : tensor<1x?x1xf32>) {
    ^bb0(%in: f64, %out: f32):
      %1190 = arith.truncf %in : f64 to f32
      linalg.yield %1190 : f32
    } -> tensor<1x?x1xf32>
    %993 = linalg.generic {indexing_maps = [#map1, #map11], iterator_types = ["parallel", "parallel", "reduction"]} ins(%984 : tensor<1x?x768xf32>) outs(%960 : tensor<1x?x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1190 = arith.addf %in, %out : f32
      linalg.yield %1190 : f32
    } -> tensor<1x?x1xf32>
    %994 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%993 : tensor<1x?x1xf32>) outs(%958 : tensor<1x?x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1190 = arith.divf %in, %cst_158 : f32
      linalg.yield %1190 : f32
    } -> tensor<1x?x1xf32>
    %995 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%992 : tensor<1x?x1xf32>) outs(%958 : tensor<1x?x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1190 = arith.truncf %cst_154 : f64 to f32
      %1191 = arith.addf %in, %1190 : f32
      linalg.yield %1191 : f32
    } -> tensor<1x?x1xf32>
    %996 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%995 : tensor<1x?x1xf32>) outs(%958 : tensor<1x?x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1190 = math.rsqrt %in : f32
      linalg.yield %1190 : f32
    } -> tensor<1x?x1xf32>
    %997 = linalg.generic {indexing_maps = [#map1, #map11, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%984, %994 : tensor<1x?x768xf32>, tensor<1x?x1xf32>) outs(%946 : tensor<1x?x768xf32>) {
    ^bb0(%in: f32, %in_495: f32, %out: f32):
      %1190 = arith.subf %in, %in_495 : f32
      linalg.yield %1190 : f32
    } -> tensor<1x?x768xf32>
    %998 = linalg.generic {indexing_maps = [#map1, #map11, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%997, %996 : tensor<1x?x768xf32>, tensor<1x?x1xf32>) outs(%946 : tensor<1x?x768xf32>) {
    ^bb0(%in: f32, %in_495: f32, %out: f32):
      %1190 = arith.mulf %in, %in_495 : f32
      linalg.yield %1190 : f32
    } -> tensor<1x?x768xf32>
    %999 = linalg.generic {indexing_maps = [#map1, #map12, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%998, %cst_29 : tensor<1x?x768xf32>, tensor<768xf32>) outs(%946 : tensor<1x?x768xf32>) {
    ^bb0(%in: f32, %in_495: f32, %out: f32):
      %1190 = arith.mulf %in, %in_495 : f32
      linalg.yield %1190 : f32
    } -> tensor<1x?x768xf32>
    %1000 = linalg.generic {indexing_maps = [#map1, #map12, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%999, %cst_28 : tensor<1x?x768xf32>, tensor<768xf32>) outs(%946 : tensor<1x?x768xf32>) {
    ^bb0(%in: f32, %in_495: f32, %out: f32):
      %1190 = arith.addf %in, %in_495 : f32
      linalg.yield %1190 : f32
    } -> tensor<1x?x768xf32>
    %collapsed_436 = tensor.collapse_shape %1000 [[0, 1], [2]] : tensor<1x?x768xf32> into tensor<?x768xf32>
    %1001 = tensor.empty(%940) : tensor<?x2304xf32>
    %1002 = linalg.fill ins(%cst_0 : f32) outs(%1001 : tensor<?x2304xf32>) -> tensor<?x2304xf32>
    %1003 = linalg.matmul ins(%collapsed_436, %cst_27 : tensor<?x768xf32>, tensor<768x2304xf32>) outs(%1002 : tensor<?x2304xf32>) -> tensor<?x2304xf32>
    %1004 = linalg.generic {indexing_maps = [#map3, #map13, #map3], iterator_types = ["parallel", "parallel"]} ins(%1003, %cst_26 : tensor<?x2304xf32>, tensor<2304xf32>) outs(%1001 : tensor<?x2304xf32>) {
    ^bb0(%in: f32, %in_495: f32, %out: f32):
      %1190 = arith.addf %in, %in_495 : f32
      linalg.yield %1190 : f32
    } -> tensor<?x2304xf32>
    %expanded_437 = tensor.expand_shape %1004 [[0, 1], [2]] output_shape [1, %940, 2304] : tensor<?x2304xf32> into tensor<1x?x2304xf32>
    %extracted_slice_438 = tensor.extract_slice %expanded_437[0, 0, 0] [1, %940, 768] [1, 1, 1] : tensor<1x?x2304xf32> to tensor<1x?x768xf32>
    %extracted_slice_439 = tensor.extract_slice %expanded_437[0, 0, 768] [1, %940, 768] [1, 1, 1] : tensor<1x?x2304xf32> to tensor<1x?x768xf32>
    %extracted_slice_440 = tensor.extract_slice %expanded_437[0, 0, 1536] [1, %940, 768] [1, 1, 1] : tensor<1x?x2304xf32> to tensor<1x?x768xf32>
    cf.assert %62, "must have at most one inferred (negative) dimension"
    %1005 = arith.muli %129, %c768_i64 : i64
    %1006 = arith.divsi %1005, %61 : i64
    %1007 = arith.select %1, %1006, %0 : i64
    %from_elements_441 = tensor.from_elements %c1_i64, %1007, %1006, %c64_i64 : tensor<4xi64>
    %reshape_442 = tensor.reshape %extracted_slice_439(%from_elements_441) : (tensor<1x?x768xf32>, tensor<4xi64>) -> tensor<1x?x12x64xf32>
    %1008 = arith.index_cast %1007 : i64 to index
    cf.assert %62, "must have at most one inferred (negative) dimension"
    %reshape_443 = tensor.reshape %extracted_slice_440(%from_elements_441) : (tensor<1x?x768xf32>, tensor<4xi64>) -> tensor<1x?x12x64xf32>
    %1009 = tensor.empty(%1008) : tensor<1x12x?x64xf32>
    %transposed_444 = linalg.transpose ins(%reshape_443 : tensor<1x?x12x64xf32>) outs(%1009 : tensor<1x12x?x64xf32>) permutation = [0, 2, 1, 3] 
    cf.assert %62, "must have at most one inferred (negative) dimension"
    %reshape_445 = tensor.reshape %extracted_slice_438(%from_elements_441) : (tensor<1x?x768xf32>, tensor<4xi64>) -> tensor<1x?x12x64xf32>
    %transposed_446 = linalg.transpose ins(%reshape_445 : tensor<1x?x12x64xf32>) outs(%1009 : tensor<1x12x?x64xf32>) permutation = [0, 2, 1, 3] 
    %1010 = linalg.generic {indexing_maps = [#map7, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%transposed_446 : tensor<1x12x?x64xf32>) outs(%1009 : tensor<1x12x?x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1190 = arith.truncf %cst_148 : f64 to f32
      %1191 = arith.mulf %in, %1190 : f32
      linalg.yield %1191 : f32
    } -> tensor<1x12x?x64xf32>
    %1011 = tensor.empty(%1008) : tensor<1x12x64x?xf32>
    %transposed_447 = linalg.transpose ins(%reshape_442 : tensor<1x?x12x64xf32>) outs(%1011 : tensor<1x12x64x?xf32>) permutation = [0, 2, 3, 1] 
    %1012 = linalg.generic {indexing_maps = [#map7, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%transposed_447 : tensor<1x12x64x?xf32>) outs(%1011 : tensor<1x12x64x?xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1190 = arith.truncf %cst_148 : f64 to f32
      %1191 = arith.mulf %in, %1190 : f32
      linalg.yield %1191 : f32
    } -> tensor<1x12x64x?xf32>
    cf.assert %30, "unimplemented: dynamic negative broadcast sizes"
    %1013 = linalg.generic {indexing_maps = [#map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%73 : tensor<1x12x?x64xf32>) {
    ^bb0(%out: f32):
      %1190 = linalg.index 1 : index
      %1191 = linalg.index 2 : index
      %1192 = linalg.index 3 : index
      %1193 = arith.cmpi eq, %1008, %c1 : index
      %1194 = arith.select %1193, %c0, %1191 : index
      %extracted = tensor.extract %1010[%c0, %1190, %1194, %1192] : tensor<1x12x?x64xf32>
      linalg.yield %extracted : f32
    } -> tensor<1x12x?x64xf32>
    %collapsed_448 = tensor.collapse_shape %1013 [[0, 1], [2], [3]] : tensor<1x12x?x64xf32> into tensor<12x?x64xf32>
    cf.assert %30, "unimplemented: dynamic negative broadcast sizes"
    %1014 = linalg.generic {indexing_maps = [#map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%75 : tensor<1x12x64x?xf32>) {
    ^bb0(%out: f32):
      %1190 = linalg.index 1 : index
      %1191 = linalg.index 2 : index
      %1192 = linalg.index 3 : index
      %1193 = arith.cmpi eq, %1008, %c1 : index
      %1194 = arith.select %1193, %c0, %1192 : index
      %extracted = tensor.extract %1012[%c0, %1190, %1191, %1194] : tensor<1x12x64x?xf32>
      linalg.yield %extracted : f32
    } -> tensor<1x12x64x?xf32>
    %collapsed_449 = tensor.collapse_shape %1014 [[0, 1], [2], [3]] : tensor<1x12x64x?xf32> into tensor<12x64x?xf32>
    %1015 = linalg.batch_matmul ins(%collapsed_448, %collapsed_449 : tensor<12x?x64xf32>, tensor<12x64x?xf32>) outs(%78 : tensor<12x?x?xf32>) -> tensor<12x?x?xf32>
    cf.assert %85, "must have at most one inferred (negative) dimension"
    %reshape_450 = tensor.reshape %1015(%from_elements_177) : (tensor<12x?x?xf32>, tensor<4xi64>) -> tensor<1x12x?x?xf32>
    cf.assert %90, "mismatched size for broadcast"
    cf.assert %90, "mismatched size for broadcast"
    %1016 = linalg.generic {indexing_maps = [#map7, #map14, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%reshape_450, %69 : tensor<1x12x?x?xf32>, tensor<1x1x?x?xf32>) outs(%91 : tensor<1x12x?x?xf32>) {
    ^bb0(%in: f32, %in_495: f32, %out: f32):
      %1190 = arith.addf %in, %in_495 : f32
      linalg.yield %1190 : f32
    } -> tensor<1x12x?x?xf32>
    %1017:2 = linalg.generic {indexing_maps = [#map7, #map15, #map15], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%1016 : tensor<1x12x?x?xf32>) outs(%96, %94 : tensor<1x12x?xf32>, tensor<1x12x?xi64>) {
    ^bb0(%in: f32, %out: f32, %out_495: i64):
      %1190 = linalg.index 3 : index
      %1191 = arith.index_cast %1190 : index to i64
      %1192 = arith.maximumf %in, %out : f32
      %1193 = arith.cmpf ogt, %in, %out : f32
      %1194 = arith.select %1193, %1191, %out_495 : i64
      linalg.yield %1192, %1194 : f32, i64
    } -> (tensor<1x12x?xf32>, tensor<1x12x?xi64>)
    %expanded_451 = tensor.expand_shape %1017#0 [[0], [1], [2, 3]] output_shape [1, 12, %89, 1] : tensor<1x12x?xf32> into tensor<1x12x?x1xf32>
    %1018 = linalg.generic {indexing_maps = [#map7, #map6, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1016, %expanded_451 : tensor<1x12x?x?xf32>, tensor<1x12x?x1xf32>) outs(%91 : tensor<1x12x?x?xf32>) {
    ^bb0(%in: f32, %in_495: f32, %out: f32):
      %1190 = arith.subf %in, %in_495 : f32
      linalg.yield %1190 : f32
    } -> tensor<1x12x?x?xf32>
    %1019 = linalg.generic {indexing_maps = [#map7, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1018 : tensor<1x12x?x?xf32>) outs(%91 : tensor<1x12x?x?xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1190 = math.exp %in : f32
      linalg.yield %1190 : f32
    } -> tensor<1x12x?x?xf32>
    %1020 = linalg.generic {indexing_maps = [#map7, #map6], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%1019 : tensor<1x12x?x?xf32>) outs(%101 : tensor<1x12x?x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1190 = arith.addf %in, %out : f32
      linalg.yield %1190 : f32
    } -> tensor<1x12x?x1xf32>
    %1021 = linalg.generic {indexing_maps = [#map7, #map6, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1019, %1020 : tensor<1x12x?x?xf32>, tensor<1x12x?x1xf32>) outs(%91 : tensor<1x12x?x?xf32>) {
    ^bb0(%in: f32, %in_495: f32, %out: f32):
      %1190 = arith.divf %in, %in_495 : f32
      linalg.yield %1190 : f32
    } -> tensor<1x12x?x?xf32>
    %1022 = linalg.generic {indexing_maps = [#map7, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1016 : tensor<1x12x?x?xf32>) outs(%104 : tensor<1x12x?x?xi1>) {
    ^bb0(%in: f32, %out: i1):
      %1190 = arith.extf %in : f32 to f64
      %1191 = arith.cmpf oeq, %1190, %cst_149 : f64
      linalg.yield %1191 : i1
    } -> tensor<1x12x?x?xi1>
    %1023 = linalg.generic {indexing_maps = [#map7, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1022 : tensor<1x12x?x?xi1>) outs(%104 : tensor<1x12x?x?xi1>) {
    ^bb0(%in: i1, %out: i1):
      %1190 = arith.cmpi eq, %in, %false : i1
      linalg.yield %1190 : i1
    } -> tensor<1x12x?x?xi1>
    %1024 = linalg.generic {indexing_maps = [#map7, #map6], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%1023 : tensor<1x12x?x?xi1>) outs(%108 : tensor<1x12x?x1xi1>) {
    ^bb0(%in: i1, %out: i1):
      %1190 = arith.ori %in, %out : i1
      linalg.yield %1190 : i1
    } -> tensor<1x12x?x1xi1>
    %1025 = linalg.generic {indexing_maps = [#map7, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1024 : tensor<1x12x?x1xi1>) outs(%107 : tensor<1x12x?x1xi1>) {
    ^bb0(%in: i1, %out: i1):
      %1190 = arith.cmpi eq, %in, %false : i1
      linalg.yield %1190 : i1
    } -> tensor<1x12x?x1xi1>
    cf.assert %111, "negative values not allowed in new dimensions"
    cf.assert %111, "negative values not allowed in new dimensions"
    cf.assert %115, "mismatched size for broadcast"
    cf.assert %116, "mismatched size for broadcast"
    %1026 = linalg.generic {indexing_maps = [#map6, #map7, #map7, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1025, %114, %1021 : tensor<1x12x?x1xi1>, tensor<1x12x?x?xf32>, tensor<1x12x?x?xf32>) outs(%117 : tensor<1x12x?x?xf32>) {
    ^bb0(%in: i1, %in_495: f32, %in_496: f32, %out: f32):
      %1190 = arith.select %in, %in_495, %in_496 : f32
      linalg.yield %1190 : f32
    } -> tensor<1x12x?x?xf32>
    cf.assert %30, "unimplemented: dynamic negative broadcast sizes"
    cf.assert %30, "unimplemented: dynamic negative broadcast sizes"
    %1027 = linalg.generic {indexing_maps = [#map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%119 : tensor<1x12x?x?xf32>) {
    ^bb0(%out: f32):
      %1190 = linalg.index 1 : index
      %1191 = linalg.index 2 : index
      %1192 = linalg.index 3 : index
      %1193 = arith.cmpi eq, %89, %c1 : index
      %1194 = arith.select %1193, %c0, %1191 : index
      %1195 = arith.cmpi eq, %112, %c1 : index
      %1196 = arith.select %1195, %c0, %1192 : index
      %extracted = tensor.extract %1026[%c0, %1190, %1194, %1196] : tensor<1x12x?x?xf32>
      linalg.yield %extracted : f32
    } -> tensor<1x12x?x?xf32>
    cf.assert %85, "must have at most one inferred (negative) dimension"
    %reshape_452 = tensor.reshape %1027(%from_elements_180) : (tensor<1x12x?x?xf32>, tensor<3xi64>) -> tensor<12x?x?xf32>
    cf.assert %30, "unimplemented: dynamic negative broadcast sizes"
    %1028 = linalg.generic {indexing_maps = [#map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%73 : tensor<1x12x?x64xf32>) {
    ^bb0(%out: f32):
      %1190 = linalg.index 1 : index
      %1191 = linalg.index 2 : index
      %1192 = linalg.index 3 : index
      %1193 = arith.cmpi eq, %1008, %c1 : index
      %1194 = arith.select %1193, %c0, %1191 : index
      %extracted = tensor.extract %transposed_444[%c0, %1190, %1194, %1192] : tensor<1x12x?x64xf32>
      linalg.yield %extracted : f32
    } -> tensor<1x12x?x64xf32>
    %collapsed_453 = tensor.collapse_shape %1028 [[0, 1], [2], [3]] : tensor<1x12x?x64xf32> into tensor<12x?x64xf32>
    cf.assert %122, "mismatching contracting dimension"
    %1029 = linalg.batch_matmul ins(%reshape_452, %collapsed_453 : tensor<12x?x?xf32>, tensor<12x?x64xf32>) outs(%124 : tensor<12x?x64xf32>) -> tensor<12x?x64xf32>
    %expanded_454 = tensor.expand_shape %1029 [[0, 1], [2], [3]] output_shape [1, 12, %89, 64] : tensor<12x?x64xf32> into tensor<1x12x?x64xf32>
    %transposed_455 = linalg.transpose ins(%expanded_454 : tensor<1x12x?x64xf32>) outs(%126 : tensor<1x?x12x64xf32>) permutation = [0, 2, 1, 3] 
    cf.assert %62, "must have at most one inferred (negative) dimension"
    %reshape_456 = tensor.reshape %transposed_455(%from_elements_185) : (tensor<1x?x12x64xf32>, tensor<3xi64>) -> tensor<1x?x768xf32>
    %collapsed_457 = tensor.collapse_shape %reshape_456 [[0, 1], [2]] : tensor<1x?x768xf32> into tensor<?x768xf32>
    %1030 = arith.index_cast %129 : i64 to index
    %1031 = tensor.empty(%1030) : tensor<?x768xf32>
    %1032 = linalg.fill ins(%cst_0 : f32) outs(%1031 : tensor<?x768xf32>) -> tensor<?x768xf32>
    %1033 = linalg.matmul ins(%collapsed_457, %cst_25 : tensor<?x768xf32>, tensor<768x768xf32>) outs(%1032 : tensor<?x768xf32>) -> tensor<?x768xf32>
    %1034 = linalg.generic {indexing_maps = [#map3, #map13, #map3], iterator_types = ["parallel", "parallel"]} ins(%1033, %cst_24 : tensor<?x768xf32>, tensor<768xf32>) outs(%1031 : tensor<?x768xf32>) {
    ^bb0(%in: f32, %in_495: f32, %out: f32):
      %1190 = arith.addf %in, %in_495 : f32
      linalg.yield %1190 : f32
    } -> tensor<?x768xf32>
    %expanded_458 = tensor.expand_shape %1034 [[0, 1], [2]] output_shape [1, %1030, 768] : tensor<?x768xf32> into tensor<1x?x768xf32>
    %1035 = arith.cmpi eq, %1030, %940 : index
    cf.assert %1035, "mismatched size for broadcast"
    %1036 = tensor.empty(%1030) : tensor<1x?x768xf32>
    %1037 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%expanded_458, %984 : tensor<1x?x768xf32>, tensor<1x?x768xf32>) outs(%1036 : tensor<1x?x768xf32>) {
    ^bb0(%in: f32, %in_495: f32, %out: f32):
      %1190 = arith.addf %in, %in_495 : f32
      linalg.yield %1190 : f32
    } -> tensor<1x?x768xf32>
    %1038 = tensor.empty(%1030) : tensor<1x?x768xf64>
    %1039 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1037 : tensor<1x?x768xf32>) outs(%1038 : tensor<1x?x768xf64>) {
    ^bb0(%in: f32, %out: f64):
      %1190 = arith.extf %in : f32 to f64
      linalg.yield %1190 : f64
    } -> tensor<1x?x768xf64>
    %1040 = tensor.empty(%1030) : tensor<1x?x1xf64>
    %1041 = linalg.fill ins(%cst : f64) outs(%1040 : tensor<1x?x1xf64>) -> tensor<1x?x1xf64>
    %1042 = linalg.generic {indexing_maps = [#map1, #map11], iterator_types = ["parallel", "parallel", "reduction"]} ins(%1039 : tensor<1x?x768xf64>) outs(%1041 : tensor<1x?x1xf64>) {
    ^bb0(%in: f64, %out: f64):
      %1190 = arith.addf %in, %out : f64
      linalg.yield %1190 : f64
    } -> tensor<1x?x1xf64>
    %1043 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1042 : tensor<1x?x1xf64>) outs(%1040 : tensor<1x?x1xf64>) {
    ^bb0(%in: f64, %out: f64):
      %1190 = arith.divf %in, %cst_157 : f64
      linalg.yield %1190 : f64
    } -> tensor<1x?x1xf64>
    %1044 = linalg.generic {indexing_maps = [#map1, #map11, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1039, %1043 : tensor<1x?x768xf64>, tensor<1x?x1xf64>) outs(%1038 : tensor<1x?x768xf64>) {
    ^bb0(%in: f64, %in_495: f64, %out: f64):
      %1190 = arith.subf %in, %in_495 : f64
      linalg.yield %1190 : f64
    } -> tensor<1x?x768xf64>
    %1045 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1044, %1044 : tensor<1x?x768xf64>, tensor<1x?x768xf64>) outs(%1038 : tensor<1x?x768xf64>) {
    ^bb0(%in: f64, %in_495: f64, %out: f64):
      %1190 = arith.mulf %in, %in_495 : f64
      linalg.yield %1190 : f64
    } -> tensor<1x?x768xf64>
    %1046 = linalg.generic {indexing_maps = [#map1, #map11], iterator_types = ["parallel", "parallel", "reduction"]} ins(%1045 : tensor<1x?x768xf64>) outs(%1041 : tensor<1x?x1xf64>) {
    ^bb0(%in: f64, %out: f64):
      %1190 = arith.addf %in, %out : f64
      linalg.yield %1190 : f64
    } -> tensor<1x?x1xf64>
    %1047 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1046 : tensor<1x?x1xf64>) outs(%1040 : tensor<1x?x1xf64>) {
    ^bb0(%in: f64, %out: f64):
      %1190 = arith.divf %in, %cst_157 : f64
      linalg.yield %1190 : f64
    } -> tensor<1x?x1xf64>
    %1048 = tensor.empty(%1030) : tensor<1x?x1xf32>
    %1049 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1047 : tensor<1x?x1xf64>) outs(%1048 : tensor<1x?x1xf32>) {
    ^bb0(%in: f64, %out: f32):
      %1190 = arith.truncf %in : f64 to f32
      linalg.yield %1190 : f32
    } -> tensor<1x?x1xf32>
    %1050 = linalg.fill ins(%cst_0 : f32) outs(%1048 : tensor<1x?x1xf32>) -> tensor<1x?x1xf32>
    %1051 = linalg.generic {indexing_maps = [#map1, #map11], iterator_types = ["parallel", "parallel", "reduction"]} ins(%1037 : tensor<1x?x768xf32>) outs(%1050 : tensor<1x?x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1190 = arith.addf %in, %out : f32
      linalg.yield %1190 : f32
    } -> tensor<1x?x1xf32>
    %1052 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1051 : tensor<1x?x1xf32>) outs(%1048 : tensor<1x?x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1190 = arith.divf %in, %cst_158 : f32
      linalg.yield %1190 : f32
    } -> tensor<1x?x1xf32>
    %1053 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1049 : tensor<1x?x1xf32>) outs(%1048 : tensor<1x?x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1190 = arith.truncf %cst_154 : f64 to f32
      %1191 = arith.addf %in, %1190 : f32
      linalg.yield %1191 : f32
    } -> tensor<1x?x1xf32>
    %1054 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1053 : tensor<1x?x1xf32>) outs(%1048 : tensor<1x?x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1190 = math.rsqrt %in : f32
      linalg.yield %1190 : f32
    } -> tensor<1x?x1xf32>
    %1055 = linalg.generic {indexing_maps = [#map1, #map11, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1037, %1052 : tensor<1x?x768xf32>, tensor<1x?x1xf32>) outs(%1036 : tensor<1x?x768xf32>) {
    ^bb0(%in: f32, %in_495: f32, %out: f32):
      %1190 = arith.subf %in, %in_495 : f32
      linalg.yield %1190 : f32
    } -> tensor<1x?x768xf32>
    %1056 = linalg.generic {indexing_maps = [#map1, #map11, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1055, %1054 : tensor<1x?x768xf32>, tensor<1x?x1xf32>) outs(%1036 : tensor<1x?x768xf32>) {
    ^bb0(%in: f32, %in_495: f32, %out: f32):
      %1190 = arith.mulf %in, %in_495 : f32
      linalg.yield %1190 : f32
    } -> tensor<1x?x768xf32>
    %1057 = linalg.generic {indexing_maps = [#map1, #map12, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1056, %cst_23 : tensor<1x?x768xf32>, tensor<768xf32>) outs(%1036 : tensor<1x?x768xf32>) {
    ^bb0(%in: f32, %in_495: f32, %out: f32):
      %1190 = arith.mulf %in, %in_495 : f32
      linalg.yield %1190 : f32
    } -> tensor<1x?x768xf32>
    %1058 = linalg.generic {indexing_maps = [#map1, #map12, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1057, %cst_22 : tensor<1x?x768xf32>, tensor<768xf32>) outs(%1036 : tensor<1x?x768xf32>) {
    ^bb0(%in: f32, %in_495: f32, %out: f32):
      %1190 = arith.addf %in, %in_495 : f32
      linalg.yield %1190 : f32
    } -> tensor<1x?x768xf32>
    %collapsed_459 = tensor.collapse_shape %1058 [[0, 1], [2]] : tensor<1x?x768xf32> into tensor<?x768xf32>
    %1059 = tensor.empty(%1030) : tensor<?x3072xf32>
    %1060 = linalg.fill ins(%cst_0 : f32) outs(%1059 : tensor<?x3072xf32>) -> tensor<?x3072xf32>
    %1061 = linalg.matmul ins(%collapsed_459, %cst_21 : tensor<?x768xf32>, tensor<768x3072xf32>) outs(%1060 : tensor<?x3072xf32>) -> tensor<?x3072xf32>
    %1062 = linalg.generic {indexing_maps = [#map3, #map13, #map3], iterator_types = ["parallel", "parallel"]} ins(%1061, %cst_20 : tensor<?x3072xf32>, tensor<3072xf32>) outs(%1059 : tensor<?x3072xf32>) {
    ^bb0(%in: f32, %in_495: f32, %out: f32):
      %1190 = arith.addf %in, %in_495 : f32
      linalg.yield %1190 : f32
    } -> tensor<?x3072xf32>
    %expanded_460 = tensor.expand_shape %1062 [[0, 1], [2]] output_shape [1, %1030, 3072] : tensor<?x3072xf32> into tensor<1x?x3072xf32>
    %1063 = tensor.empty(%1030) : tensor<1x?x3072xf32>
    %1064 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%expanded_460 : tensor<1x?x3072xf32>) outs(%1063 : tensor<1x?x3072xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1190 = arith.mulf %in, %cst_159 : f32
      linalg.yield %1190 : f32
    } -> tensor<1x?x3072xf32>
    %1065 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%expanded_460 : tensor<1x?x3072xf32>) outs(%1063 : tensor<1x?x3072xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1190 = math.fpowi %in, %c3_i64 : f32, i64
      linalg.yield %1190 : f32
    } -> tensor<1x?x3072xf32>
    %1066 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1065 : tensor<1x?x3072xf32>) outs(%1063 : tensor<1x?x3072xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1190 = arith.truncf %cst_141 : f64 to f32
      %1191 = arith.mulf %in, %1190 : f32
      linalg.yield %1191 : f32
    } -> tensor<1x?x3072xf32>
    %1067 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%expanded_460, %1066 : tensor<1x?x3072xf32>, tensor<1x?x3072xf32>) outs(%1063 : tensor<1x?x3072xf32>) {
    ^bb0(%in: f32, %in_495: f32, %out: f32):
      %1190 = arith.addf %in, %in_495 : f32
      linalg.yield %1190 : f32
    } -> tensor<1x?x3072xf32>
    %1068 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1067 : tensor<1x?x3072xf32>) outs(%1063 : tensor<1x?x3072xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1190 = arith.truncf %cst_140 : f64 to f32
      %1191 = arith.mulf %in, %1190 : f32
      linalg.yield %1191 : f32
    } -> tensor<1x?x3072xf32>
    %1069 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1068 : tensor<1x?x3072xf32>) outs(%1063 : tensor<1x?x3072xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1190 = math.tanh %in : f32
      linalg.yield %1190 : f32
    } -> tensor<1x?x3072xf32>
    %1070 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1069 : tensor<1x?x3072xf32>) outs(%1063 : tensor<1x?x3072xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1190 = arith.addf %in, %cst_160 : f32
      linalg.yield %1190 : f32
    } -> tensor<1x?x3072xf32>
    %1071 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1064, %1070 : tensor<1x?x3072xf32>, tensor<1x?x3072xf32>) outs(%1063 : tensor<1x?x3072xf32>) {
    ^bb0(%in: f32, %in_495: f32, %out: f32):
      %1190 = arith.mulf %in, %in_495 : f32
      linalg.yield %1190 : f32
    } -> tensor<1x?x3072xf32>
    %collapsed_461 = tensor.collapse_shape %1071 [[0, 1], [2]] : tensor<1x?x3072xf32> into tensor<?x3072xf32>
    %1072 = linalg.matmul ins(%collapsed_461, %cst_19 : tensor<?x3072xf32>, tensor<3072x768xf32>) outs(%1032 : tensor<?x768xf32>) -> tensor<?x768xf32>
    %1073 = linalg.generic {indexing_maps = [#map3, #map13, #map3], iterator_types = ["parallel", "parallel"]} ins(%1072, %cst_18 : tensor<?x768xf32>, tensor<768xf32>) outs(%1031 : tensor<?x768xf32>) {
    ^bb0(%in: f32, %in_495: f32, %out: f32):
      %1190 = arith.addf %in, %in_495 : f32
      linalg.yield %1190 : f32
    } -> tensor<?x768xf32>
    %expanded_462 = tensor.expand_shape %1073 [[0, 1], [2]] output_shape [1, %1030, 768] : tensor<?x768xf32> into tensor<1x?x768xf32>
    %1074 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1037, %expanded_462 : tensor<1x?x768xf32>, tensor<1x?x768xf32>) outs(%1036 : tensor<1x?x768xf32>) {
    ^bb0(%in: f32, %in_495: f32, %out: f32):
      %1190 = arith.addf %in, %in_495 : f32
      linalg.yield %1190 : f32
    } -> tensor<1x?x768xf32>
    %1075 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1074 : tensor<1x?x768xf32>) outs(%1038 : tensor<1x?x768xf64>) {
    ^bb0(%in: f32, %out: f64):
      %1190 = arith.extf %in : f32 to f64
      linalg.yield %1190 : f64
    } -> tensor<1x?x768xf64>
    %1076 = linalg.generic {indexing_maps = [#map1, #map11], iterator_types = ["parallel", "parallel", "reduction"]} ins(%1075 : tensor<1x?x768xf64>) outs(%1041 : tensor<1x?x1xf64>) {
    ^bb0(%in: f64, %out: f64):
      %1190 = arith.addf %in, %out : f64
      linalg.yield %1190 : f64
    } -> tensor<1x?x1xf64>
    %1077 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1076 : tensor<1x?x1xf64>) outs(%1040 : tensor<1x?x1xf64>) {
    ^bb0(%in: f64, %out: f64):
      %1190 = arith.divf %in, %cst_157 : f64
      linalg.yield %1190 : f64
    } -> tensor<1x?x1xf64>
    %1078 = linalg.generic {indexing_maps = [#map1, #map11, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1075, %1077 : tensor<1x?x768xf64>, tensor<1x?x1xf64>) outs(%1038 : tensor<1x?x768xf64>) {
    ^bb0(%in: f64, %in_495: f64, %out: f64):
      %1190 = arith.subf %in, %in_495 : f64
      linalg.yield %1190 : f64
    } -> tensor<1x?x768xf64>
    %1079 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1078, %1078 : tensor<1x?x768xf64>, tensor<1x?x768xf64>) outs(%1038 : tensor<1x?x768xf64>) {
    ^bb0(%in: f64, %in_495: f64, %out: f64):
      %1190 = arith.mulf %in, %in_495 : f64
      linalg.yield %1190 : f64
    } -> tensor<1x?x768xf64>
    %1080 = linalg.generic {indexing_maps = [#map1, #map11], iterator_types = ["parallel", "parallel", "reduction"]} ins(%1079 : tensor<1x?x768xf64>) outs(%1041 : tensor<1x?x1xf64>) {
    ^bb0(%in: f64, %out: f64):
      %1190 = arith.addf %in, %out : f64
      linalg.yield %1190 : f64
    } -> tensor<1x?x1xf64>
    %1081 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1080 : tensor<1x?x1xf64>) outs(%1040 : tensor<1x?x1xf64>) {
    ^bb0(%in: f64, %out: f64):
      %1190 = arith.divf %in, %cst_157 : f64
      linalg.yield %1190 : f64
    } -> tensor<1x?x1xf64>
    %1082 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1081 : tensor<1x?x1xf64>) outs(%1048 : tensor<1x?x1xf32>) {
    ^bb0(%in: f64, %out: f32):
      %1190 = arith.truncf %in : f64 to f32
      linalg.yield %1190 : f32
    } -> tensor<1x?x1xf32>
    %1083 = linalg.generic {indexing_maps = [#map1, #map11], iterator_types = ["parallel", "parallel", "reduction"]} ins(%1074 : tensor<1x?x768xf32>) outs(%1050 : tensor<1x?x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1190 = arith.addf %in, %out : f32
      linalg.yield %1190 : f32
    } -> tensor<1x?x1xf32>
    %1084 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1083 : tensor<1x?x1xf32>) outs(%1048 : tensor<1x?x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1190 = arith.divf %in, %cst_158 : f32
      linalg.yield %1190 : f32
    } -> tensor<1x?x1xf32>
    %1085 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1082 : tensor<1x?x1xf32>) outs(%1048 : tensor<1x?x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1190 = arith.truncf %cst_154 : f64 to f32
      %1191 = arith.addf %in, %1190 : f32
      linalg.yield %1191 : f32
    } -> tensor<1x?x1xf32>
    %1086 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1085 : tensor<1x?x1xf32>) outs(%1048 : tensor<1x?x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1190 = math.rsqrt %in : f32
      linalg.yield %1190 : f32
    } -> tensor<1x?x1xf32>
    %1087 = linalg.generic {indexing_maps = [#map1, #map11, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1074, %1084 : tensor<1x?x768xf32>, tensor<1x?x1xf32>) outs(%1036 : tensor<1x?x768xf32>) {
    ^bb0(%in: f32, %in_495: f32, %out: f32):
      %1190 = arith.subf %in, %in_495 : f32
      linalg.yield %1190 : f32
    } -> tensor<1x?x768xf32>
    %1088 = linalg.generic {indexing_maps = [#map1, #map11, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1087, %1086 : tensor<1x?x768xf32>, tensor<1x?x1xf32>) outs(%1036 : tensor<1x?x768xf32>) {
    ^bb0(%in: f32, %in_495: f32, %out: f32):
      %1190 = arith.mulf %in, %in_495 : f32
      linalg.yield %1190 : f32
    } -> tensor<1x?x768xf32>
    %1089 = linalg.generic {indexing_maps = [#map1, #map12, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1088, %cst_17 : tensor<1x?x768xf32>, tensor<768xf32>) outs(%1036 : tensor<1x?x768xf32>) {
    ^bb0(%in: f32, %in_495: f32, %out: f32):
      %1190 = arith.mulf %in, %in_495 : f32
      linalg.yield %1190 : f32
    } -> tensor<1x?x768xf32>
    %1090 = linalg.generic {indexing_maps = [#map1, #map12, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1089, %cst_16 : tensor<1x?x768xf32>, tensor<768xf32>) outs(%1036 : tensor<1x?x768xf32>) {
    ^bb0(%in: f32, %in_495: f32, %out: f32):
      %1190 = arith.addf %in, %in_495 : f32
      linalg.yield %1190 : f32
    } -> tensor<1x?x768xf32>
    %collapsed_463 = tensor.collapse_shape %1090 [[0, 1], [2]] : tensor<1x?x768xf32> into tensor<?x768xf32>
    %1091 = tensor.empty(%1030) : tensor<?x2304xf32>
    %1092 = linalg.fill ins(%cst_0 : f32) outs(%1091 : tensor<?x2304xf32>) -> tensor<?x2304xf32>
    %1093 = linalg.matmul ins(%collapsed_463, %cst_15 : tensor<?x768xf32>, tensor<768x2304xf32>) outs(%1092 : tensor<?x2304xf32>) -> tensor<?x2304xf32>
    %1094 = linalg.generic {indexing_maps = [#map3, #map13, #map3], iterator_types = ["parallel", "parallel"]} ins(%1093, %cst_14 : tensor<?x2304xf32>, tensor<2304xf32>) outs(%1091 : tensor<?x2304xf32>) {
    ^bb0(%in: f32, %in_495: f32, %out: f32):
      %1190 = arith.addf %in, %in_495 : f32
      linalg.yield %1190 : f32
    } -> tensor<?x2304xf32>
    %expanded_464 = tensor.expand_shape %1094 [[0, 1], [2]] output_shape [1, %1030, 2304] : tensor<?x2304xf32> into tensor<1x?x2304xf32>
    %extracted_slice_465 = tensor.extract_slice %expanded_464[0, 0, 0] [1, %1030, 768] [1, 1, 1] : tensor<1x?x2304xf32> to tensor<1x?x768xf32>
    %extracted_slice_466 = tensor.extract_slice %expanded_464[0, 0, 768] [1, %1030, 768] [1, 1, 1] : tensor<1x?x2304xf32> to tensor<1x?x768xf32>
    %extracted_slice_467 = tensor.extract_slice %expanded_464[0, 0, 1536] [1, %1030, 768] [1, 1, 1] : tensor<1x?x2304xf32> to tensor<1x?x768xf32>
    cf.assert %62, "must have at most one inferred (negative) dimension"
    %1095 = arith.muli %129, %c768_i64 : i64
    %1096 = arith.divsi %1095, %61 : i64
    %1097 = arith.select %1, %1096, %0 : i64
    %from_elements_468 = tensor.from_elements %c1_i64, %1097, %1096, %c64_i64 : tensor<4xi64>
    %reshape_469 = tensor.reshape %extracted_slice_466(%from_elements_468) : (tensor<1x?x768xf32>, tensor<4xi64>) -> tensor<1x?x12x64xf32>
    %1098 = arith.index_cast %1097 : i64 to index
    cf.assert %62, "must have at most one inferred (negative) dimension"
    %reshape_470 = tensor.reshape %extracted_slice_467(%from_elements_468) : (tensor<1x?x768xf32>, tensor<4xi64>) -> tensor<1x?x12x64xf32>
    %1099 = tensor.empty(%1098) : tensor<1x12x?x64xf32>
    %transposed_471 = linalg.transpose ins(%reshape_470 : tensor<1x?x12x64xf32>) outs(%1099 : tensor<1x12x?x64xf32>) permutation = [0, 2, 1, 3] 
    cf.assert %62, "must have at most one inferred (negative) dimension"
    %reshape_472 = tensor.reshape %extracted_slice_465(%from_elements_468) : (tensor<1x?x768xf32>, tensor<4xi64>) -> tensor<1x?x12x64xf32>
    %transposed_473 = linalg.transpose ins(%reshape_472 : tensor<1x?x12x64xf32>) outs(%1099 : tensor<1x12x?x64xf32>) permutation = [0, 2, 1, 3] 
    %1100 = linalg.generic {indexing_maps = [#map7, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%transposed_473 : tensor<1x12x?x64xf32>) outs(%1099 : tensor<1x12x?x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1190 = arith.truncf %cst_148 : f64 to f32
      %1191 = arith.mulf %in, %1190 : f32
      linalg.yield %1191 : f32
    } -> tensor<1x12x?x64xf32>
    %1101 = tensor.empty(%1098) : tensor<1x12x64x?xf32>
    %transposed_474 = linalg.transpose ins(%reshape_469 : tensor<1x?x12x64xf32>) outs(%1101 : tensor<1x12x64x?xf32>) permutation = [0, 2, 3, 1] 
    %1102 = linalg.generic {indexing_maps = [#map7, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%transposed_474 : tensor<1x12x64x?xf32>) outs(%1101 : tensor<1x12x64x?xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1190 = arith.truncf %cst_148 : f64 to f32
      %1191 = arith.mulf %in, %1190 : f32
      linalg.yield %1191 : f32
    } -> tensor<1x12x64x?xf32>
    cf.assert %30, "unimplemented: dynamic negative broadcast sizes"
    %1103 = linalg.generic {indexing_maps = [#map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%73 : tensor<1x12x?x64xf32>) {
    ^bb0(%out: f32):
      %1190 = linalg.index 1 : index
      %1191 = linalg.index 2 : index
      %1192 = linalg.index 3 : index
      %1193 = arith.cmpi eq, %1098, %c1 : index
      %1194 = arith.select %1193, %c0, %1191 : index
      %extracted = tensor.extract %1100[%c0, %1190, %1194, %1192] : tensor<1x12x?x64xf32>
      linalg.yield %extracted : f32
    } -> tensor<1x12x?x64xf32>
    %collapsed_475 = tensor.collapse_shape %1103 [[0, 1], [2], [3]] : tensor<1x12x?x64xf32> into tensor<12x?x64xf32>
    cf.assert %30, "unimplemented: dynamic negative broadcast sizes"
    %1104 = linalg.generic {indexing_maps = [#map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%75 : tensor<1x12x64x?xf32>) {
    ^bb0(%out: f32):
      %1190 = linalg.index 1 : index
      %1191 = linalg.index 2 : index
      %1192 = linalg.index 3 : index
      %1193 = arith.cmpi eq, %1098, %c1 : index
      %1194 = arith.select %1193, %c0, %1192 : index
      %extracted = tensor.extract %1102[%c0, %1190, %1191, %1194] : tensor<1x12x64x?xf32>
      linalg.yield %extracted : f32
    } -> tensor<1x12x64x?xf32>
    %collapsed_476 = tensor.collapse_shape %1104 [[0, 1], [2], [3]] : tensor<1x12x64x?xf32> into tensor<12x64x?xf32>
    %1105 = linalg.batch_matmul ins(%collapsed_475, %collapsed_476 : tensor<12x?x64xf32>, tensor<12x64x?xf32>) outs(%78 : tensor<12x?x?xf32>) -> tensor<12x?x?xf32>
    cf.assert %85, "must have at most one inferred (negative) dimension"
    %reshape_477 = tensor.reshape %1105(%from_elements_177) : (tensor<12x?x?xf32>, tensor<4xi64>) -> tensor<1x12x?x?xf32>
    cf.assert %90, "mismatched size for broadcast"
    cf.assert %90, "mismatched size for broadcast"
    %1106 = linalg.generic {indexing_maps = [#map7, #map14, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%reshape_477, %69 : tensor<1x12x?x?xf32>, tensor<1x1x?x?xf32>) outs(%91 : tensor<1x12x?x?xf32>) {
    ^bb0(%in: f32, %in_495: f32, %out: f32):
      %1190 = arith.addf %in, %in_495 : f32
      linalg.yield %1190 : f32
    } -> tensor<1x12x?x?xf32>
    %1107:2 = linalg.generic {indexing_maps = [#map7, #map15, #map15], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%1106 : tensor<1x12x?x?xf32>) outs(%96, %94 : tensor<1x12x?xf32>, tensor<1x12x?xi64>) {
    ^bb0(%in: f32, %out: f32, %out_495: i64):
      %1190 = linalg.index 3 : index
      %1191 = arith.index_cast %1190 : index to i64
      %1192 = arith.maximumf %in, %out : f32
      %1193 = arith.cmpf ogt, %in, %out : f32
      %1194 = arith.select %1193, %1191, %out_495 : i64
      linalg.yield %1192, %1194 : f32, i64
    } -> (tensor<1x12x?xf32>, tensor<1x12x?xi64>)
    %expanded_478 = tensor.expand_shape %1107#0 [[0], [1], [2, 3]] output_shape [1, 12, %89, 1] : tensor<1x12x?xf32> into tensor<1x12x?x1xf32>
    %1108 = linalg.generic {indexing_maps = [#map7, #map6, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1106, %expanded_478 : tensor<1x12x?x?xf32>, tensor<1x12x?x1xf32>) outs(%91 : tensor<1x12x?x?xf32>) {
    ^bb0(%in: f32, %in_495: f32, %out: f32):
      %1190 = arith.subf %in, %in_495 : f32
      linalg.yield %1190 : f32
    } -> tensor<1x12x?x?xf32>
    %1109 = linalg.generic {indexing_maps = [#map7, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1108 : tensor<1x12x?x?xf32>) outs(%91 : tensor<1x12x?x?xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1190 = math.exp %in : f32
      linalg.yield %1190 : f32
    } -> tensor<1x12x?x?xf32>
    %1110 = linalg.generic {indexing_maps = [#map7, #map6], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%1109 : tensor<1x12x?x?xf32>) outs(%101 : tensor<1x12x?x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1190 = arith.addf %in, %out : f32
      linalg.yield %1190 : f32
    } -> tensor<1x12x?x1xf32>
    %1111 = linalg.generic {indexing_maps = [#map7, #map6, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1109, %1110 : tensor<1x12x?x?xf32>, tensor<1x12x?x1xf32>) outs(%91 : tensor<1x12x?x?xf32>) {
    ^bb0(%in: f32, %in_495: f32, %out: f32):
      %1190 = arith.divf %in, %in_495 : f32
      linalg.yield %1190 : f32
    } -> tensor<1x12x?x?xf32>
    %1112 = linalg.generic {indexing_maps = [#map7, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1106 : tensor<1x12x?x?xf32>) outs(%104 : tensor<1x12x?x?xi1>) {
    ^bb0(%in: f32, %out: i1):
      %1190 = arith.extf %in : f32 to f64
      %1191 = arith.cmpf oeq, %1190, %cst_149 : f64
      linalg.yield %1191 : i1
    } -> tensor<1x12x?x?xi1>
    %1113 = linalg.generic {indexing_maps = [#map7, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1112 : tensor<1x12x?x?xi1>) outs(%104 : tensor<1x12x?x?xi1>) {
    ^bb0(%in: i1, %out: i1):
      %1190 = arith.cmpi eq, %in, %false : i1
      linalg.yield %1190 : i1
    } -> tensor<1x12x?x?xi1>
    %1114 = linalg.generic {indexing_maps = [#map7, #map6], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%1113 : tensor<1x12x?x?xi1>) outs(%108 : tensor<1x12x?x1xi1>) {
    ^bb0(%in: i1, %out: i1):
      %1190 = arith.ori %in, %out : i1
      linalg.yield %1190 : i1
    } -> tensor<1x12x?x1xi1>
    %1115 = linalg.generic {indexing_maps = [#map7, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1114 : tensor<1x12x?x1xi1>) outs(%107 : tensor<1x12x?x1xi1>) {
    ^bb0(%in: i1, %out: i1):
      %1190 = arith.cmpi eq, %in, %false : i1
      linalg.yield %1190 : i1
    } -> tensor<1x12x?x1xi1>
    cf.assert %111, "negative values not allowed in new dimensions"
    cf.assert %111, "negative values not allowed in new dimensions"
    cf.assert %115, "mismatched size for broadcast"
    cf.assert %116, "mismatched size for broadcast"
    %1116 = linalg.generic {indexing_maps = [#map6, #map7, #map7, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1115, %114, %1111 : tensor<1x12x?x1xi1>, tensor<1x12x?x?xf32>, tensor<1x12x?x?xf32>) outs(%117 : tensor<1x12x?x?xf32>) {
    ^bb0(%in: i1, %in_495: f32, %in_496: f32, %out: f32):
      %1190 = arith.select %in, %in_495, %in_496 : f32
      linalg.yield %1190 : f32
    } -> tensor<1x12x?x?xf32>
    cf.assert %30, "unimplemented: dynamic negative broadcast sizes"
    cf.assert %30, "unimplemented: dynamic negative broadcast sizes"
    %1117 = linalg.generic {indexing_maps = [#map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%119 : tensor<1x12x?x?xf32>) {
    ^bb0(%out: f32):
      %1190 = linalg.index 1 : index
      %1191 = linalg.index 2 : index
      %1192 = linalg.index 3 : index
      %1193 = arith.cmpi eq, %89, %c1 : index
      %1194 = arith.select %1193, %c0, %1191 : index
      %1195 = arith.cmpi eq, %112, %c1 : index
      %1196 = arith.select %1195, %c0, %1192 : index
      %extracted = tensor.extract %1116[%c0, %1190, %1194, %1196] : tensor<1x12x?x?xf32>
      linalg.yield %extracted : f32
    } -> tensor<1x12x?x?xf32>
    cf.assert %85, "must have at most one inferred (negative) dimension"
    %reshape_479 = tensor.reshape %1117(%from_elements_180) : (tensor<1x12x?x?xf32>, tensor<3xi64>) -> tensor<12x?x?xf32>
    cf.assert %30, "unimplemented: dynamic negative broadcast sizes"
    %1118 = linalg.generic {indexing_maps = [#map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%73 : tensor<1x12x?x64xf32>) {
    ^bb0(%out: f32):
      %1190 = linalg.index 1 : index
      %1191 = linalg.index 2 : index
      %1192 = linalg.index 3 : index
      %1193 = arith.cmpi eq, %1098, %c1 : index
      %1194 = arith.select %1193, %c0, %1191 : index
      %extracted = tensor.extract %transposed_471[%c0, %1190, %1194, %1192] : tensor<1x12x?x64xf32>
      linalg.yield %extracted : f32
    } -> tensor<1x12x?x64xf32>
    %collapsed_480 = tensor.collapse_shape %1118 [[0, 1], [2], [3]] : tensor<1x12x?x64xf32> into tensor<12x?x64xf32>
    cf.assert %122, "mismatching contracting dimension"
    %1119 = linalg.batch_matmul ins(%reshape_479, %collapsed_480 : tensor<12x?x?xf32>, tensor<12x?x64xf32>) outs(%124 : tensor<12x?x64xf32>) -> tensor<12x?x64xf32>
    %expanded_481 = tensor.expand_shape %1119 [[0, 1], [2], [3]] output_shape [1, 12, %89, 64] : tensor<12x?x64xf32> into tensor<1x12x?x64xf32>
    %transposed_482 = linalg.transpose ins(%expanded_481 : tensor<1x12x?x64xf32>) outs(%126 : tensor<1x?x12x64xf32>) permutation = [0, 2, 1, 3] 
    cf.assert %62, "must have at most one inferred (negative) dimension"
    %reshape_483 = tensor.reshape %transposed_482(%from_elements_185) : (tensor<1x?x12x64xf32>, tensor<3xi64>) -> tensor<1x?x768xf32>
    %collapsed_484 = tensor.collapse_shape %reshape_483 [[0, 1], [2]] : tensor<1x?x768xf32> into tensor<?x768xf32>
    %1120 = arith.index_cast %129 : i64 to index
    %1121 = tensor.empty(%1120) : tensor<?x768xf32>
    %1122 = linalg.fill ins(%cst_0 : f32) outs(%1121 : tensor<?x768xf32>) -> tensor<?x768xf32>
    %1123 = linalg.matmul ins(%collapsed_484, %cst_13 : tensor<?x768xf32>, tensor<768x768xf32>) outs(%1122 : tensor<?x768xf32>) -> tensor<?x768xf32>
    %1124 = linalg.generic {indexing_maps = [#map3, #map13, #map3], iterator_types = ["parallel", "parallel"]} ins(%1123, %cst_12 : tensor<?x768xf32>, tensor<768xf32>) outs(%1121 : tensor<?x768xf32>) {
    ^bb0(%in: f32, %in_495: f32, %out: f32):
      %1190 = arith.addf %in, %in_495 : f32
      linalg.yield %1190 : f32
    } -> tensor<?x768xf32>
    %expanded_485 = tensor.expand_shape %1124 [[0, 1], [2]] output_shape [1, %1120, 768] : tensor<?x768xf32> into tensor<1x?x768xf32>
    %1125 = arith.cmpi eq, %1120, %1030 : index
    cf.assert %1125, "mismatched size for broadcast"
    %1126 = tensor.empty(%1120) : tensor<1x?x768xf32>
    %1127 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%expanded_485, %1074 : tensor<1x?x768xf32>, tensor<1x?x768xf32>) outs(%1126 : tensor<1x?x768xf32>) {
    ^bb0(%in: f32, %in_495: f32, %out: f32):
      %1190 = arith.addf %in, %in_495 : f32
      linalg.yield %1190 : f32
    } -> tensor<1x?x768xf32>
    %1128 = tensor.empty(%1120) : tensor<1x?x768xf64>
    %1129 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1127 : tensor<1x?x768xf32>) outs(%1128 : tensor<1x?x768xf64>) {
    ^bb0(%in: f32, %out: f64):
      %1190 = arith.extf %in : f32 to f64
      linalg.yield %1190 : f64
    } -> tensor<1x?x768xf64>
    %1130 = tensor.empty(%1120) : tensor<1x?x1xf64>
    %1131 = linalg.fill ins(%cst : f64) outs(%1130 : tensor<1x?x1xf64>) -> tensor<1x?x1xf64>
    %1132 = linalg.generic {indexing_maps = [#map1, #map11], iterator_types = ["parallel", "parallel", "reduction"]} ins(%1129 : tensor<1x?x768xf64>) outs(%1131 : tensor<1x?x1xf64>) {
    ^bb0(%in: f64, %out: f64):
      %1190 = arith.addf %in, %out : f64
      linalg.yield %1190 : f64
    } -> tensor<1x?x1xf64>
    %1133 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1132 : tensor<1x?x1xf64>) outs(%1130 : tensor<1x?x1xf64>) {
    ^bb0(%in: f64, %out: f64):
      %1190 = arith.divf %in, %cst_157 : f64
      linalg.yield %1190 : f64
    } -> tensor<1x?x1xf64>
    %1134 = linalg.generic {indexing_maps = [#map1, #map11, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1129, %1133 : tensor<1x?x768xf64>, tensor<1x?x1xf64>) outs(%1128 : tensor<1x?x768xf64>) {
    ^bb0(%in: f64, %in_495: f64, %out: f64):
      %1190 = arith.subf %in, %in_495 : f64
      linalg.yield %1190 : f64
    } -> tensor<1x?x768xf64>
    %1135 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1134, %1134 : tensor<1x?x768xf64>, tensor<1x?x768xf64>) outs(%1128 : tensor<1x?x768xf64>) {
    ^bb0(%in: f64, %in_495: f64, %out: f64):
      %1190 = arith.mulf %in, %in_495 : f64
      linalg.yield %1190 : f64
    } -> tensor<1x?x768xf64>
    %1136 = linalg.generic {indexing_maps = [#map1, #map11], iterator_types = ["parallel", "parallel", "reduction"]} ins(%1135 : tensor<1x?x768xf64>) outs(%1131 : tensor<1x?x1xf64>) {
    ^bb0(%in: f64, %out: f64):
      %1190 = arith.addf %in, %out : f64
      linalg.yield %1190 : f64
    } -> tensor<1x?x1xf64>
    %1137 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1136 : tensor<1x?x1xf64>) outs(%1130 : tensor<1x?x1xf64>) {
    ^bb0(%in: f64, %out: f64):
      %1190 = arith.divf %in, %cst_157 : f64
      linalg.yield %1190 : f64
    } -> tensor<1x?x1xf64>
    %1138 = tensor.empty(%1120) : tensor<1x?x1xf32>
    %1139 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1137 : tensor<1x?x1xf64>) outs(%1138 : tensor<1x?x1xf32>) {
    ^bb0(%in: f64, %out: f32):
      %1190 = arith.truncf %in : f64 to f32
      linalg.yield %1190 : f32
    } -> tensor<1x?x1xf32>
    %1140 = linalg.fill ins(%cst_0 : f32) outs(%1138 : tensor<1x?x1xf32>) -> tensor<1x?x1xf32>
    %1141 = linalg.generic {indexing_maps = [#map1, #map11], iterator_types = ["parallel", "parallel", "reduction"]} ins(%1127 : tensor<1x?x768xf32>) outs(%1140 : tensor<1x?x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1190 = arith.addf %in, %out : f32
      linalg.yield %1190 : f32
    } -> tensor<1x?x1xf32>
    %1142 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1141 : tensor<1x?x1xf32>) outs(%1138 : tensor<1x?x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1190 = arith.divf %in, %cst_158 : f32
      linalg.yield %1190 : f32
    } -> tensor<1x?x1xf32>
    %1143 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1139 : tensor<1x?x1xf32>) outs(%1138 : tensor<1x?x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1190 = arith.truncf %cst_154 : f64 to f32
      %1191 = arith.addf %in, %1190 : f32
      linalg.yield %1191 : f32
    } -> tensor<1x?x1xf32>
    %1144 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1143 : tensor<1x?x1xf32>) outs(%1138 : tensor<1x?x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1190 = math.rsqrt %in : f32
      linalg.yield %1190 : f32
    } -> tensor<1x?x1xf32>
    %1145 = linalg.generic {indexing_maps = [#map1, #map11, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1127, %1142 : tensor<1x?x768xf32>, tensor<1x?x1xf32>) outs(%1126 : tensor<1x?x768xf32>) {
    ^bb0(%in: f32, %in_495: f32, %out: f32):
      %1190 = arith.subf %in, %in_495 : f32
      linalg.yield %1190 : f32
    } -> tensor<1x?x768xf32>
    %1146 = linalg.generic {indexing_maps = [#map1, #map11, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1145, %1144 : tensor<1x?x768xf32>, tensor<1x?x1xf32>) outs(%1126 : tensor<1x?x768xf32>) {
    ^bb0(%in: f32, %in_495: f32, %out: f32):
      %1190 = arith.mulf %in, %in_495 : f32
      linalg.yield %1190 : f32
    } -> tensor<1x?x768xf32>
    %1147 = linalg.generic {indexing_maps = [#map1, #map12, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1146, %cst_11 : tensor<1x?x768xf32>, tensor<768xf32>) outs(%1126 : tensor<1x?x768xf32>) {
    ^bb0(%in: f32, %in_495: f32, %out: f32):
      %1190 = arith.mulf %in, %in_495 : f32
      linalg.yield %1190 : f32
    } -> tensor<1x?x768xf32>
    %1148 = linalg.generic {indexing_maps = [#map1, #map12, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1147, %cst_10 : tensor<1x?x768xf32>, tensor<768xf32>) outs(%1126 : tensor<1x?x768xf32>) {
    ^bb0(%in: f32, %in_495: f32, %out: f32):
      %1190 = arith.addf %in, %in_495 : f32
      linalg.yield %1190 : f32
    } -> tensor<1x?x768xf32>
    %collapsed_486 = tensor.collapse_shape %1148 [[0, 1], [2]] : tensor<1x?x768xf32> into tensor<?x768xf32>
    %1149 = tensor.empty(%1120) : tensor<?x3072xf32>
    %1150 = linalg.fill ins(%cst_0 : f32) outs(%1149 : tensor<?x3072xf32>) -> tensor<?x3072xf32>
    %1151 = linalg.matmul ins(%collapsed_486, %cst_9 : tensor<?x768xf32>, tensor<768x3072xf32>) outs(%1150 : tensor<?x3072xf32>) -> tensor<?x3072xf32>
    %1152 = linalg.generic {indexing_maps = [#map3, #map13, #map3], iterator_types = ["parallel", "parallel"]} ins(%1151, %cst_8 : tensor<?x3072xf32>, tensor<3072xf32>) outs(%1149 : tensor<?x3072xf32>) {
    ^bb0(%in: f32, %in_495: f32, %out: f32):
      %1190 = arith.addf %in, %in_495 : f32
      linalg.yield %1190 : f32
    } -> tensor<?x3072xf32>
    %expanded_487 = tensor.expand_shape %1152 [[0, 1], [2]] output_shape [1, %1120, 3072] : tensor<?x3072xf32> into tensor<1x?x3072xf32>
    %1153 = tensor.empty(%1120) : tensor<1x?x3072xf32>
    %1154 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%expanded_487 : tensor<1x?x3072xf32>) outs(%1153 : tensor<1x?x3072xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1190 = arith.mulf %in, %cst_159 : f32
      linalg.yield %1190 : f32
    } -> tensor<1x?x3072xf32>
    %1155 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%expanded_487 : tensor<1x?x3072xf32>) outs(%1153 : tensor<1x?x3072xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1190 = math.fpowi %in, %c3_i64 : f32, i64
      linalg.yield %1190 : f32
    } -> tensor<1x?x3072xf32>
    %1156 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1155 : tensor<1x?x3072xf32>) outs(%1153 : tensor<1x?x3072xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1190 = arith.truncf %cst_141 : f64 to f32
      %1191 = arith.mulf %in, %1190 : f32
      linalg.yield %1191 : f32
    } -> tensor<1x?x3072xf32>
    %1157 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%expanded_487, %1156 : tensor<1x?x3072xf32>, tensor<1x?x3072xf32>) outs(%1153 : tensor<1x?x3072xf32>) {
    ^bb0(%in: f32, %in_495: f32, %out: f32):
      %1190 = arith.addf %in, %in_495 : f32
      linalg.yield %1190 : f32
    } -> tensor<1x?x3072xf32>
    %1158 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1157 : tensor<1x?x3072xf32>) outs(%1153 : tensor<1x?x3072xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1190 = arith.truncf %cst_140 : f64 to f32
      %1191 = arith.mulf %in, %1190 : f32
      linalg.yield %1191 : f32
    } -> tensor<1x?x3072xf32>
    %1159 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1158 : tensor<1x?x3072xf32>) outs(%1153 : tensor<1x?x3072xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1190 = math.tanh %in : f32
      linalg.yield %1190 : f32
    } -> tensor<1x?x3072xf32>
    %1160 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1159 : tensor<1x?x3072xf32>) outs(%1153 : tensor<1x?x3072xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1190 = arith.addf %in, %cst_160 : f32
      linalg.yield %1190 : f32
    } -> tensor<1x?x3072xf32>
    %1161 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1154, %1160 : tensor<1x?x3072xf32>, tensor<1x?x3072xf32>) outs(%1153 : tensor<1x?x3072xf32>) {
    ^bb0(%in: f32, %in_495: f32, %out: f32):
      %1190 = arith.mulf %in, %in_495 : f32
      linalg.yield %1190 : f32
    } -> tensor<1x?x3072xf32>
    %collapsed_488 = tensor.collapse_shape %1161 [[0, 1], [2]] : tensor<1x?x3072xf32> into tensor<?x3072xf32>
    %1162 = linalg.matmul ins(%collapsed_488, %cst_7 : tensor<?x3072xf32>, tensor<3072x768xf32>) outs(%1122 : tensor<?x768xf32>) -> tensor<?x768xf32>
    %1163 = linalg.generic {indexing_maps = [#map3, #map13, #map3], iterator_types = ["parallel", "parallel"]} ins(%1162, %cst_6 : tensor<?x768xf32>, tensor<768xf32>) outs(%1121 : tensor<?x768xf32>) {
    ^bb0(%in: f32, %in_495: f32, %out: f32):
      %1190 = arith.addf %in, %in_495 : f32
      linalg.yield %1190 : f32
    } -> tensor<?x768xf32>
    %expanded_489 = tensor.expand_shape %1163 [[0, 1], [2]] output_shape [1, %1120, 768] : tensor<?x768xf32> into tensor<1x?x768xf32>
    %1164 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1127, %expanded_489 : tensor<1x?x768xf32>, tensor<1x?x768xf32>) outs(%1126 : tensor<1x?x768xf32>) {
    ^bb0(%in: f32, %in_495: f32, %out: f32):
      %1190 = arith.addf %in, %in_495 : f32
      linalg.yield %1190 : f32
    } -> tensor<1x?x768xf32>
    %1165 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1164 : tensor<1x?x768xf32>) outs(%1128 : tensor<1x?x768xf64>) {
    ^bb0(%in: f32, %out: f64):
      %1190 = arith.extf %in : f32 to f64
      linalg.yield %1190 : f64
    } -> tensor<1x?x768xf64>
    %1166 = linalg.generic {indexing_maps = [#map1, #map11], iterator_types = ["parallel", "parallel", "reduction"]} ins(%1165 : tensor<1x?x768xf64>) outs(%1131 : tensor<1x?x1xf64>) {
    ^bb0(%in: f64, %out: f64):
      %1190 = arith.addf %in, %out : f64
      linalg.yield %1190 : f64
    } -> tensor<1x?x1xf64>
    %1167 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1166 : tensor<1x?x1xf64>) outs(%1130 : tensor<1x?x1xf64>) {
    ^bb0(%in: f64, %out: f64):
      %1190 = arith.divf %in, %cst_157 : f64
      linalg.yield %1190 : f64
    } -> tensor<1x?x1xf64>
    %1168 = linalg.generic {indexing_maps = [#map1, #map11, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1165, %1167 : tensor<1x?x768xf64>, tensor<1x?x1xf64>) outs(%1128 : tensor<1x?x768xf64>) {
    ^bb0(%in: f64, %in_495: f64, %out: f64):
      %1190 = arith.subf %in, %in_495 : f64
      linalg.yield %1190 : f64
    } -> tensor<1x?x768xf64>
    %1169 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1168, %1168 : tensor<1x?x768xf64>, tensor<1x?x768xf64>) outs(%1128 : tensor<1x?x768xf64>) {
    ^bb0(%in: f64, %in_495: f64, %out: f64):
      %1190 = arith.mulf %in, %in_495 : f64
      linalg.yield %1190 : f64
    } -> tensor<1x?x768xf64>
    %1170 = linalg.generic {indexing_maps = [#map1, #map11], iterator_types = ["parallel", "parallel", "reduction"]} ins(%1169 : tensor<1x?x768xf64>) outs(%1131 : tensor<1x?x1xf64>) {
    ^bb0(%in: f64, %out: f64):
      %1190 = arith.addf %in, %out : f64
      linalg.yield %1190 : f64
    } -> tensor<1x?x1xf64>
    %1171 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1170 : tensor<1x?x1xf64>) outs(%1130 : tensor<1x?x1xf64>) {
    ^bb0(%in: f64, %out: f64):
      %1190 = arith.divf %in, %cst_157 : f64
      linalg.yield %1190 : f64
    } -> tensor<1x?x1xf64>
    %1172 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1171 : tensor<1x?x1xf64>) outs(%1138 : tensor<1x?x1xf32>) {
    ^bb0(%in: f64, %out: f32):
      %1190 = arith.truncf %in : f64 to f32
      linalg.yield %1190 : f32
    } -> tensor<1x?x1xf32>
    %1173 = linalg.generic {indexing_maps = [#map1, #map11], iterator_types = ["parallel", "parallel", "reduction"]} ins(%1164 : tensor<1x?x768xf32>) outs(%1140 : tensor<1x?x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1190 = arith.addf %in, %out : f32
      linalg.yield %1190 : f32
    } -> tensor<1x?x1xf32>
    %1174 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1173 : tensor<1x?x1xf32>) outs(%1138 : tensor<1x?x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1190 = arith.divf %in, %cst_158 : f32
      linalg.yield %1190 : f32
    } -> tensor<1x?x1xf32>
    %1175 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1172 : tensor<1x?x1xf32>) outs(%1138 : tensor<1x?x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1190 = arith.truncf %cst_154 : f64 to f32
      %1191 = arith.addf %in, %1190 : f32
      linalg.yield %1191 : f32
    } -> tensor<1x?x1xf32>
    %1176 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1175 : tensor<1x?x1xf32>) outs(%1138 : tensor<1x?x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1190 = math.rsqrt %in : f32
      linalg.yield %1190 : f32
    } -> tensor<1x?x1xf32>
    %1177 = linalg.generic {indexing_maps = [#map1, #map11, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1164, %1174 : tensor<1x?x768xf32>, tensor<1x?x1xf32>) outs(%1126 : tensor<1x?x768xf32>) {
    ^bb0(%in: f32, %in_495: f32, %out: f32):
      %1190 = arith.subf %in, %in_495 : f32
      linalg.yield %1190 : f32
    } -> tensor<1x?x768xf32>
    %1178 = linalg.generic {indexing_maps = [#map1, #map11, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1177, %1176 : tensor<1x?x768xf32>, tensor<1x?x1xf32>) outs(%1126 : tensor<1x?x768xf32>) {
    ^bb0(%in: f32, %in_495: f32, %out: f32):
      %1190 = arith.mulf %in, %in_495 : f32
      linalg.yield %1190 : f32
    } -> tensor<1x?x768xf32>
    %1179 = linalg.generic {indexing_maps = [#map1, #map12, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1178, %cst_5 : tensor<1x?x768xf32>, tensor<768xf32>) outs(%1126 : tensor<1x?x768xf32>) {
    ^bb0(%in: f32, %in_495: f32, %out: f32):
      %1190 = arith.mulf %in, %in_495 : f32
      linalg.yield %1190 : f32
    } -> tensor<1x?x768xf32>
    %1180 = linalg.generic {indexing_maps = [#map1, #map12, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1179, %cst_4 : tensor<1x?x768xf32>, tensor<768xf32>) outs(%1126 : tensor<1x?x768xf32>) {
    ^bb0(%in: f32, %in_495: f32, %out: f32):
      %1190 = arith.addf %in, %in_495 : f32
      linalg.yield %1190 : f32
    } -> tensor<1x?x768xf32>
    %1181 = arith.muli %2, %c768_i64 : i64
    cf.assert %4, "must have at most one inferred (negative) dimension"
    %1182 = arith.muli %129, %c768_i64 : i64
    %1183 = arith.divsi %1182, %1181 : i64
    %1184 = arith.select %1, %1183, %0 : i64
    %from_elements_490 = tensor.from_elements %1183, %1184, %c768_i64 : tensor<3xi64>
    %reshape_491 = tensor.reshape %1180(%from_elements_490) : (tensor<1x?x768xf32>, tensor<3xi64>) -> tensor<1x?x768xf32>
    %1185 = tensor.empty() : tensor<768x50257xf32>
    %transposed_492 = linalg.transpose ins(%cst_156 : tensor<50257x768xf32>) outs(%1185 : tensor<768x50257xf32>) permutation = [1, 0] 
    %collapsed_493 = tensor.collapse_shape %reshape_491 [[0, 1], [2]] : tensor<1x?x768xf32> into tensor<?x768xf32>
    %1186 = arith.index_cast %1184 : i64 to index
    %1187 = tensor.empty(%1186) : tensor<?x50257xf32>
    %1188 = linalg.fill ins(%cst_0 : f32) outs(%1187 : tensor<?x50257xf32>) -> tensor<?x50257xf32>
    %1189 = linalg.matmul ins(%collapsed_493, %transposed_492 : tensor<?x768xf32>, tensor<768x50257xf32>) outs(%1188 : tensor<?x50257xf32>) -> tensor<?x50257xf32>
    %expanded_494 = tensor.expand_shape %1189 [[0, 1], [2]] output_shape [1, %1186, 50257] : tensor<?x50257xf32> into tensor<1x?x50257xf32>
    return %expanded_494 : tensor<1x?x50257xf32>
  }
}
