# Capstone Stackoverflow Data Prepration

## Objective

1. Download all the questions that are answered about PyTorch.
2. Take care of the code parsing and storing slightly differently such that it can later integrate with others.

## Solution
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://githubtocolab.com/pankaj90382/END-1.0/blob/main/Capstone_Stackoverflow/Capstone_test.ipynb)


### Approach

For this [question](https://stackoverflow.com/questions/64837376/how-to-efficiently-run-multiple-pytorch-processes-models-at-once-traceback) as a example, the stackoverflow api the data in different formats. The Stackoverflow api fetches the questions data with various important attributes.

```text
{'backoff': 0,
 'has_more': False,
 'items': [{'answer_count': 2,
            'body': '<p><strong>Background</strong></p>\n'
                    '<p>I have a very small network which I want to test with '
                    'different random seeds.\n'
                    'The network barely uses 1% of my GPUs compute power so i '
                    'could in theory run 50 processes at once to try many '
                    'different seeds at once.</p>\n'
                    '<p><strong>Problem</strong></p>\n'
                    "<p>Unfortunately i can't even import pytorch in multiple "
                    'processes. When the <strong>nr of processes</strong> '
                    'exceeds <strong>4</strong> I get a Traceback regarding a '
                    'too small paging file.</p>\n'
                    '<p><strong>Minimal reproducable code§ - '
                    'dispatcher.py</strong></p>\n'
                    '<pre><code>from subprocess import Popen\n'
                    'import sys\n'
                    '\n'
                    'procs = []\n'
                    'for seed in range(50):\n'
                    '    procs.append(Popen([sys.executable, '
                    '&quot;ml_model.py&quot;, str(seed)]))\n'
                    '\n'
                    'for proc in procs:\n'
                    '    proc.wait()\n'
                    '</code></pre>\n'
                    '<p>§I increased the number of seeds so people with better '
                    'machines can also reproduce this.</p>\n'
                    '<p><strong>Minimal reproducable code - '
                    'ml_model.py</strong></p>\n'
                    '<pre><code>import torch\n'
                    'import time\n'
                    'time.sleep(10)\n'
                    '</code></pre>\n'
                    '<pre><code> \n'
                    ' Traceback (most recent call last):\n'
                    '  File &quot;ml_model.py&quot;, line 1, in '
                    '&lt;module&gt;\n'
                    '    import torch\n'
                    '  File '
                    '&quot;C:\\Users\\user\\AppData\\Local\\Programs\\Python\\Python38\\lib\\site-packages\\torch\\__init__.py&quot;, '
                    'line 117, in &lt;module&gt;\n'
                    '    import torch\n'
                    '  File '
                    '&quot;C:\\Users\\user\\AppData\\Local\\Programs\\Python\\Python38\\lib\\site-packages\\torch\\__init__.py&quot;, '
                    'line 117, in &lt;module&gt;\n'
                    '    raise err\n'
                    ' OSError: [WinError 1455] The paging file is too small '
                    'for this operation to complete. Error loading '
                    '&quot;C:\\Users\\user\\AppData\\Local\\Programs\\Python\\Python38\\lib\\site-packages\\torch\\lib\\cudnn_cnn_infer64_8.dll&quot; '
                    'or one of its dependencies.\n'
                    '    raise err\n'
                    '\n'
                    '</code></pre>\n'
                    '<p><strong>Further Investigation</strong></p>\n'
                    "<p>I noticed that each process loads a lot of dll's into "
                    'RAM. And when i close all other programs which use a lot '
                    'of RAM i can get up to 10 procesess instead of 4. So it '
                    'seems like a resource constraint.</p>\n'
                    '<p><strong>Questions</strong></p>\n'
                    '<p>Is there a workaround ?</p>\n'
                    "<p>What's the recommended way to train many small "
                    'networks with pytorch on a single gpu ?</p>\n'
                    '<p>Should i write my own CUDA Kernel instead, or use a '
                    'different framework to achieve this ?</p>\n'
                    '<p>My goal would be to run around 50 processes at once '
                    '(on a 16GB RAM Machine, 8GB GPU RAM)</p>\n',
            'content_license': 'CC BY-SA 4.0',
            'creation_date': 1605379342,
            'is_answered': True,
            'last_activity_date': 1616753005,
            'link': 'https://stackoverflow.com/questions/64837376/how-to-efficiently-run-multiple-pytorch-processes-models-at-once-traceback',
            'owner': {'display_name': 'KoKlA',
                      'link': 'https://stackoverflow.com/users/5130800/kokla',
                      'profile_image': 'https://www.gravatar.com/avatar/1497fd8c4f126845db3774aab86f3058?s=128&d=identicon&r=PG&f=1',
                      'reputation': 597,
                      'user_id': 5130800,
                      'user_type': 'registered'},
            'question_id': 64837376,
            'score': 10,
            'tags': ['python', 'pytorch', 'python-multiprocessing'],
            'title': 'How to efficiently run multiple Pytorch Processes / '
                     'Models at once ? Traceback: The paging file is too small '
                     'for this operation to complete',
            'view_count': 3348}],
 'page': 1,
 'quota_max': 300,
 'quota_remaining': 298,
 'total': 0}
```

As the same question have two answers, the Stackoverflow api fetches both answers of question with their attributes seperately into different dictonaries.

```text
{'backoff': 0,
 'has_more': False,
 'items': [{'answer_id': 66814746,
            'body': '<p>For my case system is already set to system managed '
                    'size, yet I have same error, that is because I pass a big '
                    'sized variable to multiple processes within a function. '
                    'Likely I need to set a very large paging file as Windows '
                    'cannot create it on the fly, but instead opt out to '
                    'reduce number of processes as it is not an always to be '
                    'used function.</p>\n'
                    '<p>If you are in Windows it may be better to use 1 (or '
                    'more) core less than total number of <strong>pysical '
                    'cores</strong> as multiprocessing module in python in '
                    'Windows tends to get everything as possible if you use '
                    'all and actually tries to get all '
                    '<strong>logical</strong> cores.</p>\n'
                    '<pre><code>import multiprocessing\n'
                    'multiprocessing.cpu_count()\n'
                    '12  \n'
                    '# I actually have 6 pysical cores, if you use this as '
                    'base it will likely hog system\n'
                    '\n'
                    '\n'
                    'import psutil \n'
                    'psutil.cpu_count(logical = False)\n'
                    '6 #actual number of pysical cores\n'
                    '\n'
                    'psutil.cpu_count(logical = True)\n'
                    '12 #logical cores (e.g. hyperthreading)\n'
                    '</code></pre>\n'
                    '<p>Please refer to here for more detail:\n'
                    '<a '
                    'href="https://stackoverflow.com/questions/40217873/multiprocessing-use-only-the-physical-cores">Multiprocessing: '
                    'use only the physical cores?</a></p>\n',
            'content_license': 'CC BY-SA 4.0',
            'creation_date': 1616753005,
            'is_accepted': False,
            'last_activity_date': 1616753005,
            'owner': {'display_name': 'Gorkem',
                      'link': 'https://stackoverflow.com/users/1562772/gorkem',
                      'profile_image': 'https://i.stack.imgur.com/l6KDA.png?s=128&g=1',
                      'reputation': 342,
                      'user_id': 1562772,
                      'user_type': 'registered'},
            'question_id': 64837376,
            'score': 2},
           {'answer_id': 66296034,
            'body': '<p>Well, i managed to resolve this.\n'
                    'open &quot;advanced system setting&quot;. Go to the '
                    'advanced tab then click settings related to performance.\n'
                    'Again click on advanced tab--&gt; change --&gt; unselect '
                    "'automatically......'. for all the drives, set 'system "
                    "managed size'. Restart your pc.</p>\n",
            'content_license': 'CC BY-SA 4.0',
            'creation_date': 1613851468,
            'is_accepted': False,
            'last_activity_date': 1613851468,
            'owner': {'display_name': 'Tufail Waris',
                      'link': 'https://stackoverflow.com/users/11325381/tufail-waris',
                      'profile_image': 'https://lh5.googleusercontent.com/-5RDbnSkNvF8/AAAAAAAAAAI/AAAAAAAAAEk/2R5rdHLtTy0/photo.jpg?sz=128',
                      'reputation': 59,
                      'user_id': 11325381,
                      'user_type': 'registered'},
            'question_id': 64837376,
            'score': 1}],
 'page': 1,
 'quota_max': 300,
 'quota_remaining': 148,
 'total': 0}
```

To manipulate and store the data easily, I have converted the data into pandas dataframe and stored all the questions data into [seperate excel file](https://github.com/pankaj90382/END-1.0/blob/main/Capstone_Stackoverflow/Answered_Pytorch_questions_stackoverflow_with_body.xlsx) which represents all the attributes releated to the questions. As for one question, there will be multiple answers so it is better to store all the answers to [store seperately](https://github.com/pankaj90382/END-1.0/blob/main/Capstone_Stackoverflow/Answered_Pytorch_answers_stackoverflow_with_body.xlsx) with all answer attributes. To get full picture of the question and answers, there is need to perform the join in between two sets of data using the pandas dataframe. 


### Statistics

| Date | Pytorch Tagged Stackoverflow Questions  | Answered Pytorch Questions |
|--|--|--|
| 05th August, 2021 | 13372 | 7322 |

### Files

| Type | Files Link  |
|--|--|
| All Pytorch Questions | [![Excel](https://shields.io/badge/-Download-217346?logo=microsoft-excel&style=flat)](https://github.com/pankaj90382/END-1.0/blob/main/Capstone_Stackoverflow/Pytorch_questions_stackoverflow.xlsx) |
| Answered Pytorch Questions and Id's | [![Excel](https://shields.io/badge/-Download-217346?logo=microsoft-excel&style=flat)](https://github.com/pankaj90382/END-1.0/blob/main/Capstone_Stackoverflow/Answered_Pytorch_questions_stackoverflow.xlsx) |
| Answered Pytorch Questions with description | [![Excel](https://shields.io/badge/-Download-217346?logo=microsoft-excel&style=flat)](https://github.com/pankaj90382/END-1.0/blob/main/Capstone_Stackoverflow/Answered_Pytorch_questions_stackoverflow_with_body.xlsx) |
| Answers of Pytorch Questions with Solution | [![Excel](https://shields.io/badge/-Download-217346?logo=microsoft-excel&style=flat)](https://github.com/pankaj90382/END-1.0/blob/main/Capstone_Stackoverflow/Answered_Pytorch_answers_stackoverflow_with_body.xlsx) |

## Refrences

  - [Stackoverflow api](https://stackapi.readthedocs.io/en/latest/user/quickstart.html)
