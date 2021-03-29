# Curriculum

Modify `params/stog_amr_2.0.yaml`

- Before
```yaml

data:
  data_dir: *amr_data_dir
  train_data: train.txt.features.preproc
  dev_data: dev.txt.features.preproc
  test_data: test.txt.features.preproc
  data_type: AMR
  batch_first: True
  iterator:
    train_batch_size: &train_batch_size 64
    test_batch_size: 32
    iter_type: BucketIterator
    sorting_keys:  # (field, padding type)
      - [tgt_tokens, num_tokens]
  pretrain_token_emb: *glove
  pretrain_char_emb:
  word_splitter: *word_splitter


```

- After
```yaml

data:
  data_dir: *amr_data_dir
  train_data: train.txt.features.preproc
  dev_data: dev.txt.features.preproc
  test_data: test.txt.features.preproc
  data_type: AMR
  batch_first: True
  iterator:
    train_batch_size: &train_batch_size 64
    test_batch_size: 32
    iter_type: CompetenceCurriculumIterator    # Curriculum Setting
    sorting_keys:  # (field, padding type)
      - [tgt_tokens, num_tokens]
    curriculum_kwargs:                         # Curriculum Setting
      curriculum_len: 17130                    # Curriculum Setting
      initial_competence: 0.1                  # Curriculum Setting
      slope_power: 2                           # Curriculum Setting
      damr_name: DAMRR0V2                      # Curriculum Setting
  pretrain_token_emb: *glove
  pretrain_char_emb:
  word_splitter: *word_splitter
  
```

See more details in `stog/data/iterators/curriculum_iterator.py`