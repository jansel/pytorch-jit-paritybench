import sys
_module = sys.modules[__name__]
del sys
setup = _module
tensorhive = _module
APIServer = _module
api = _module
app = _module
AppServer = _module
web = _module
authorization = _module
cli = _module
config = _module
controllers = _module
group = _module
job = _module
nodes = _module
reservation = _module
resource = _module
restriction = _module
schedule = _module
task = _module
user = _module
core = _module
InfrastructureManager = _module
SSHConnectionManager = _module
ServiceManager = _module
TensorHiveManager = _module
managers = _module
CPUMonitor = _module
GPUMonitor = _module
Monitor = _module
monitors = _module
scheduling = _module
JobSchedulingService = _module
MonitoringService = _module
ProtectionService = _module
Service = _module
UsageLoggingService = _module
services = _module
ssh = _module
task_nursery = _module
AccountCreator = _module
NvidiaSmiParser = _module
ReservationVerifier = _module
Singleton = _module
StoppableThread = _module
utils = _module
colors = _module
decorators = _module
exceptions = _module
mailer = _module
time = _module
EmailSendingBehaviour = _module
MessageSendingBehaviour = _module
ProtectionHandler = _module
SudoProcessKillingBehaviour = _module
UserProcessKillingBehaviour = _module
violation_handlers = _module
database = _module
ForbiddenException = _module
InvalidRequestException = _module
env = _module
a16bb624004f_modify_tasks_table_to_match_jobs_table = _module
a44e0949e0a0_create_jobs_table = _module
bffd7d81d326_add_summary_fields_to_reservation = _module
ce624ab2c458_create_tables = _module
e792ab930685_rename_columns_to_match_api = _module
e935d47c4cde_create_restrictions_and_secondary_tables = _module
ecd059f567b5_create_groups_and_user2group_tables = _module
CRUDModel = _module
CommandSegment = _module
Group = _module
Job = _module
Reservation = _module
Resource = _module
Restriction = _module
RestrictionAssignee = _module
RestrictionSchedule = _module
RevokedToken = _module
Role = _module
Task = _module
User = _module
models = _module
DateUtils = _module
Weekday = _module
conftest = _module
auth_patcher = _module
test_group_controller = _module
test_group_controller_superuser = _module
test_job_controller = _module
test_job_controller_superuser = _module
test_reservation_controller = _module
test_reservation_controller_superuser = _module
test_restriction_controller = _module
test_restriction_controller_superuser = _module
test_schedule_controller = _module
test_schedule_controller_superuser = _module
test_task_controller = _module
test_task_controller_superuser = _module
test_user_controller = _module
test_user_controller_superuser = _module
test_account_creator = _module
test_group_model = _module
test_job_model = _module
test_reservation_model = _module
test_resource_model = _module
test_restriction_model = _module
test_restrictionschedule_model = _module
test_user_model = _module
test_decorators = _module
test_mailbot = _module
test_ssh = _module

from _paritybench_helpers import _mock_config, patch_functional
from unittest.mock import mock_open, MagicMock
from torch.autograd import Function
from torch.nn import Module
import abc, collections, copy, enum, functools, inspect, itertools, logging, math, matplotlib, numbers, numpy, pandas, queue, random, re, scipy, sklearn, string, tensorflow, time, torch, torchaudio, torchtext, torchvision, types, typing, uuid, warnings
import numpy as np
from torch import Tensor
patch_functional()
open = mock_open()
yaml = logging = sys = argparse = MagicMock()
ArgumentParser = argparse.ArgumentParser
_global_config = args = argv = cfg = config = params = _mock_config()
argparse.ArgumentParser.return_value.parse_args.return_value = _global_config
yaml.load.return_value = _global_config
sys.argv = _global_config
__version__ = '1.0.0'
xrange = range
wraps = functools.wraps

