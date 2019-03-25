{
  'targets': [{
    'target_name': 'nimiq_miner_cuda',
    'sources': [
      'src/native/cuda/argon2d.cu',
      'src/native/cuda/blake2b.cu',
      'src/native/cuda/kernels.cu',
      'src/native/cuda/miner.cc'
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
        '-gencode', 'arch=compute_35,code=sm_35',
        '-gencode', 'arch=compute_61,code=sm_61',
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
    'cflags_cc': ['-Wall', '-O3', '-fexceptions']
  }]
}
