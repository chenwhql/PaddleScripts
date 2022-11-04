"""
model clip
"""
import paddle
import paddle.static as static


def clip_extra_program_only(orig_program_path, clipped_program_path):
    """
    load inference model(program only) and clip extra op
    Args:
        orig_program_path(str): input model path
        clipped_program_path(str): output model path
    Returns:
        None
    """
    paddle.enable_static()
    origin_program_bytes = static.io.load_from_file(orig_program_path)
    origin_program = static.io.deserialize_program(origin_program_bytes)
    print(origin_program)

    clipped_program = origin_program._remove_training_info(clip_extra=True)
    print(clipped_program)
    clipped_program_bytes = static.io._serialize_program(clipped_program)
    static.io.save_to_file(clipped_program_path, clipped_program_bytes)

if __name__ == "__main__":
    clip_extra_program_only(orig_program_path="./linear773_63.pdmodel", clipped_program_path="./clipped_linear773_63.pdmodel")