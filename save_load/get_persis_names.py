import six
import paddle
from paddle.fluid import core

def load_program_desc(model_file_path):
    # 1. parse program desc
    with open(model_file_path, "rb") as f:
        program_desc_str = f.read()

    program_desc = core.ProgramDesc(program_desc_str)
    if not core._is_program_version_supported(program_desc._version()):
        raise ValueError("Unsupported program version: %d\n" %
                         program_desc._version())

    return program_desc


def is_persistable(var_desc):
    if var_desc.type() == core.VarDesc.VarType.FEED_MINIBATCH or \
            var_desc.type() == core.VarDesc.VarType.FETCH_LIST or \
            var_desc.type() == core.VarDesc.VarType.READER or \
            var_desc.type() == core.VarDesc.VarType.RAW:
        return False
    return var_desc.persistable()


def get_persistable_vars(program_desc):
    persistable_vars = []
    for i in six.moves.range(program_desc.num_blocks()):
        block = program_desc.block(i)
        persistable_vars.extend(list(filter(is_persistable, block.all_vars())))
    return persistable_vars


def get_persistable_var_names(program_desc):
    """
    Get all persistable variable names in ProgramDesc.
    """
    var_names = []
    persistable_vars = get_persistable_vars(program_desc)
    for var in persistable_vars:
        var_names.append(var.name())
    return var_names

program_desc = load_program_desc("./dy2stat_infer_model/__model__")
var_names = get_persistable_var_names(program_desc)
print(var_names)