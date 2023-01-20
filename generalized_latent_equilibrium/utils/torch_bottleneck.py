################################################################################################################
# torch profiling adapted from https://github.com/pytorch/pytorch/blob/master/torch/utils/bottleneck/__main__.py
###
import torch
import cProfile
import pstats
import sys

from torch.autograd import profiler
from torch.utils.collect_env import get_env_info
from functools import partial


def bottleneck(profiled_function, cprofile_sortby='tottime', cprofile_topk=15, autograd_prof_sortby='cpu_time_total', autograd_prof_topk=15):
    """
    Profile given function for bottlenecks with the cprofiler and pytorch's autograd profiler.
    Args:
        profiled_function: The torch function to be profiled.
        cprofile_sortby: By what to sort the cprofile output: 'ncalls, 'tottime', 'percall', 'cumtime'
        cprofile_topk: The number of printed lines from cprofile output
        autograd_prof_sortby: By what to sort the autograd profiling output: 'cpu_time', 'cuda_time', 'cpu_time_total', 'cuda_time_total', 'count'
        autograd_prof_topk: The number of printed lines from autograd profiling output.

    Returns: cpu_profiler, cuda_profiler
    """
    env_summary = run_env_analysis()

    #################
    # profile function
    if torch.cuda.is_available():
        torch.cuda.init()
    cprofile_prof = run_cprofile(profiled_function)
    autograd_prof_cpu, autograd_prof_cuda = run_autograd_prof(profiled_function)

    #################
    # print summaries
    print(env_summary)
    print_cprofile_summary(cprofile_prof, cprofile_sortby, cprofile_topk)

    if not torch.cuda.is_available():
        print_autograd_prof_summary(autograd_prof_cpu, 'CPU', autograd_prof_sortby, autograd_prof_topk)
    else:
        # Print both the result of the CPU-mode and CUDA-mode autograd profilers
        # if their execution times are very different.
        cuda_prof_exec_time = cpu_time_total(autograd_prof_cuda)
        if len(autograd_prof_cpu.function_events) > 0:
            cpu_prof_exec_time = cpu_time_total(autograd_prof_cpu)
            pct_diff = (cuda_prof_exec_time - cpu_prof_exec_time) / cuda_prof_exec_time
            if abs(pct_diff) > 0.05:
                print_autograd_prof_summary(autograd_prof_cpu, 'CPU', autograd_prof_sortby, autograd_prof_topk)

        print_autograd_prof_summary(autograd_prof_cuda, 'CUDA', autograd_prof_sortby, autograd_prof_topk)

    return cprofile_prof, autograd_prof_cpu, autograd_prof_cuda


class autograd_bottleneck(object):

    def __init__(self,  topk=15, sortby='cpu_time', profile_store=None):
        self.sortby = sortby
        self.topk = topk
        self.is_cuda_available = torch.cuda.is_available()
        self.profile_store = profile_store

    def __call__(self, profiled_function):

        def new_profiled_function(*original_args, **original_kwargs):
            pf = partial(profiled_function, *original_args, **original_kwargs)
            self.autograd_prof = run_autograd_profiler(pf, use_cuda=self.is_cuda_available)  # profile the function with autograd
            mode = "CUDA" if self.is_cuda_available else "CPU"
            if not isinstance(self.profile_store, list):
                print_autograd_prof_summary(self.autograd_prof, mode=mode, sortby=self.sortby, topk=self.topk)
            else:
                self.profile_store.append(create_autograd_prof_summary(self.autograd_prof, mode, sortby=self.sortby, topk=self.topk))

        return new_profiled_function


def cpu_time_total(autograd_prof):
    return sum([event.cpu_time_total for event in autograd_prof.function_events])


def compiled_with_cuda(sysinfo):
    if sysinfo.cuda_compiled_version:
        return 'compiled w/ CUDA {}'.format(sysinfo.cuda_compiled_version)
    return 'not compiled w/ CUDA'


env_summary = """
--------------------------------------------------------------------------------
  Environment Summary
--------------------------------------------------------------------------------
PyTorch {pytorch_version}{debug_str} {cuda_compiled}
Running with Python {py_version} and {cuda_runtime}
`{pip_version} list` truncated output:
{pip_list_output}
""".strip()


def run_env_analysis():
    print('Running environment analysis...')
    info = get_env_info()

    debug_str = ''
    if info.is_debug_build:
        debug_str = ' DEBUG'

    cuda_avail = ''
    if info.is_cuda_available:
        cuda = info.cuda_runtime_version
        if cuda is not None:
            cuda_avail = 'CUDA ' + cuda
    else:
        cuda_avail = 'CUDA unavailable'

    pip_version = info.pip_version
    pip_list_output = info.pip_packages
    if pip_list_output is None:
        pip_list_output = 'Unable to fetch'

    result = {
        'debug_str': debug_str,
        'pytorch_version': info.torch_version,
        'cuda_compiled': compiled_with_cuda(info),
        'py_version': '{}.{}'.format(sys.version_info[0], sys.version_info[1]),
        'cuda_runtime': cuda_avail,
        'pip_version': pip_version,
        'pip_list_output': pip_list_output,
    }

    return env_summary.format(**result)


def run_cprofile(profiled_function):
    print('Running your script with cProfile')
    prof = cProfile.Profile()
    prof.enable()
    profiled_function()
    prof.disable()
    return prof


def run_autograd_prof(profiled_function):

    print('Running your script with the autograd profiler...')
    result = [run_autograd_profiler(profiled_function, use_cuda=False)]
    if torch.cuda.is_available():
        result.append(run_autograd_profiler(profiled_function, use_cuda=True))
    else:
        result.append(None)

    return result


def run_autograd_profiler(profiled_function, use_cuda=False):
    with profiler.profile(use_cuda=use_cuda) as prof:
        profiled_function()
    return prof


cprof_summary = """
--------------------------------------------------------------------------------
  cProfile output
--------------------------------------------------------------------------------
""".strip()


def print_cprofile_summary(prof, sortby='tottime', topk=40):
    result = {}

    print(cprof_summary.format(**result))

    cprofile_stats = pstats.Stats(prof).sort_stats(sortby)
    cprofile_stats.print_stats(topk)


autograd_prof_summary = """
--------------------------------------------------------------------------------
  autograd profiler output ({mode} mode)
--------------------------------------------------------------------------------
        {description}
{cuda_warning}
{output}
""".strip()


def create_autograd_prof_summary(prof, mode, sortby='cpu_time', topk=15, autograd_prof_sortby='cpu_time_total'):
    valid_sortby = ['cpu_time', 'cuda_time', 'cpu_time_total', 'cuda_time_total', 'count']
    if sortby not in valid_sortby:
        warn = ('WARNING: invalid sorting option for autograd profiler results: {}\n'
                'Expected `cpu_time`, `cpu_time_total`, or `count`. '
                'Defaulting to `cpu_time`.')
        print(warn.format(autograd_prof_sortby))
        sortby = 'cpu_time'

    if mode == 'CUDA':
        cuda_warning = ('\n\tBecause the autograd profiler uses the CUDA event API,\n'
                        '\tthe CUDA time column reports approximately max(cuda_time, cpu_time).\n'
                        '\tPlease ignore this output if your code does not use CUDA.\n')
    else:
        cuda_warning = ''

    sorted_events = sorted(prof.function_events,
                           key=lambda x: getattr(x, sortby), reverse=True)
    topk_events = sorted_events[:topk]

    result = {
        'mode': mode,
        'description': 'top {} events sorted by {}'.format(topk, sortby),
        'output': torch.autograd.profiler.build_table(topk_events),
        'cuda_warning': cuda_warning
    }

    return result


def print_autograd_prof_summary(prof, mode, sortby='cpu_time', topk=15, autograd_prof_sortby='cpu_time_total'):
    result = create_autograd_prof_summary(prof, mode, sortby, topk, autograd_prof_sortby)
    print(autograd_prof_summary.format(**result))


descript = """
`bottleneck` is a tool that can be used as an initial step for debugging
bottlenecks in your program.
It summarizes runs of your script with the Python profiler and PyTorch\'s
autograd profiler. Because your script will be profiled, please ensure that it
exits in a finite amount of time.
For more complicated uses of the profilers, please see
https://docs.python.org/3/library/profile.html and
https://pytorch.org/docs/master/autograd.html#profiler for more information.
""".strip()
