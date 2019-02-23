{
  'targets': [{
    'target_name': 'nimiq_cuda_miner',
    'sources': [
      'src/native/argon2d.cu',
      'src/native/blake2b.cu',
      'src/native/kernels.cu',
      'src/native/nimiq_cuda_miner.cc'
    ],
    'rules': [{
      'extension': 'cu',
      'inputs': ['<(RULE_INPUT_PATH)'],
      'outputs':[ '<(INTERMEDIATE_DIR)/<(RULE_INPUT_ROOT).o'],
      'rule_name': 'CUDA compiler',
      'process_outputs_as_sources': 1,
      'action': [
        'nvcc', '-Xcompiler', '-fpic', '-c',
        '-O3', '--ptxas-options=-v',
        '-gencode', 'arch=compute_61,code=sm_61',
        '-gencode', 'arch=compute_75,code=sm_75',
        '-o', '<@(_outputs)', '<@(_inputs)'
      ]
    }],
    'include_dirs': [
      '<!(node -e "require(\'nan\')")',
      '/usr/local/cuda/include'
    ],
    'libraries': [
      '-lcuda', '-lcudart'
    ],
    'library_dirs': [
      '/usr/local/cuda/lib64'
    ],
    'cflags_c': ['-Wall', '-O3', '-fexceptions']
  }]
}
