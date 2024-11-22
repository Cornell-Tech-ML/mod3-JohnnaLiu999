# MiniTorch Module 3

<img src="https://minitorch.github.io/minitorch.svg" width="50%">

* Docs: https://minitorch.github.io/

* Overview: https://minitorch.github.io/module3.html


You will need to modify `tensor_functions.py` slightly in this assignment.

* Tests:

``` 
python run_tests.py
```

* Note:

Several of the tests for this assignment will only run if you are on a GPU machine and will not
run on github's test infrastructure. Please follow the instructions to setup up a colab machine
to run these tests.

This assignment requires the following files from the previous assignments. You can get these by running

```bash
python sync_previous_module.py previous-module-dir current-module-dir
```

The files that will be synced are:

        minitorch/tensor_data.py minitorch/tensor_functions.py minitorch/tensor_ops.py minitorch/operators.py minitorch/scalar.py minitorch/scalar_functions.py minitorch/module.py minitorch/autodiff.py minitorch/module.py project/run_manual.py project/run_scalar.py project/run_tensor.py minitorch/operators.py minitorch/module.py minitorch/autodiff.py minitorch/tensor.py minitorch/datasets.py minitorch/testing.py minitorch/optim.py

# Task 3.1 & 3.2
```
MAP
 
================================================================================
 Parallel Accelerator Optimizing:  Function tensor_map.<locals>._map, 
/content/mod3-JohnnaLiu999/minitorch/fast_ops.py (163)  
================================================================================


Parallel loop listing for  Function tensor_map.<locals>._map, /content/mod3-JohnnaLiu999/minitorch/fast_ops.py (163) 
--------------------------------------------------------------------------------------------|loop #ID
    def _map(                                                                               | 
        out: Storage,                                                                       | 
        out_shape: Shape,                                                                   | 
        out_strides: Strides,                                                               | 
        in_storage: Storage,                                                                | 
        in_shape: Shape,                                                                    | 
        in_strides: Strides,                                                                | 
    ) -> None:                                                                              | 
        # TODO: Implement for Task 3.1.                                                     | 
        # raise NotImplementedError("Need to implement for Task 3.1")                       | 
        # if shapes and strides are the same, we can just apply map avoid indexing          | 
        if list(out_shape) == list(in_shape) and list(out_strides) == list(in_strides):     | 
            # parallel main loop                                                            | 
            for i in prange(len(out)):------------------------------------------------------| #0
                out[i] = fn(in_storage[i])                                                  | 
        # else, we need to handle the broadcasting                                          | 
        else:                                                                               | 
            for i in prange(len(out)):------------------------------------------------------| #1
                # numpy buffers for indices                                                 | 
                in_i = np.empty(MAX_DIMS, np.int32)                                         | 
                out_i = np.empty(MAX_DIMS, np.int32)                                        | 
                to_index(i, out_shape, out_i)                                               | 
                broadcast_index(out_i, out_shape, in_shape, in_i)  # handle broadcasting    | 
                # convert back to positions                                                 | 
                in_pos = index_to_position(in_i, in_strides)                                | 
                out_pos = index_to_position(out_i, out_strides)                             | 
                out[out_pos] = fn(in_storage[in_pos])                                       | 
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 2 parallel for-
loop(s) (originating from loops labelled: #0, #1).
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel structure is already optimal.
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
 
---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
The memory allocation derived from the instruction at 
/content/mod3-JohnnaLiu999/minitorch/fast_ops.py (182) is hoisted out of the 
parallel loop labelled #1 (it will be performed before the loop is executed and 
reused inside the loop):
   Allocation:: in_i = np.empty(MAX_DIMS, np.int32)
    - numpy.empty() is used for the allocation.
The memory allocation derived from the instruction at 
/content/mod3-JohnnaLiu999/minitorch/fast_ops.py (183) is hoisted out of the 
parallel loop labelled #1 (it will be performed before the loop is executed and 
reused inside the loop):
   Allocation:: out_i = np.empty(MAX_DIMS, np.int32)
    - numpy.empty() is used for the allocation.
None
ZIP
 
================================================================================
 Parallel Accelerator Optimizing:  Function tensor_zip.<locals>._zip, 
/content/mod3-JohnnaLiu999/minitorch/fast_ops.py (217)  
================================================================================


Parallel loop listing for  Function tensor_zip.<locals>._zip, /content/mod3-JohnnaLiu999/minitorch/fast_ops.py (217) 
-----------------------------------------------------------------------------------|loop #ID
    def _zip(                                                                      | 
        out: Storage,                                                              | 
        out_shape: Shape,                                                          | 
        out_strides: Strides,                                                      | 
        a_storage: Storage,                                                        | 
        a_shape: Shape,                                                            | 
        a_strides: Strides,                                                        | 
        b_storage: Storage,                                                        | 
        b_shape: Shape,                                                            | 
        b_strides: Strides,                                                        | 
    ) -> None:                                                                     | 
        # TODO: Implement for Task 3.1.                                            | 
        # raise NotImplementedError("Need to implement for Task 3.1")              | 
        # avoid indexing if shapes and strides are the same                        | 
        if (list(out_shape) == list(a_shape) == list(b_shape)) and (               | 
            list(out_strides) == list(a_strides) == list(b_strides)                | 
        ):                                                                         | 
            for i in prange(len(out)):  # parallel main loop-----------------------| #2
                out[i] = fn(a_storage[i], b_storage[i])                            | 
        else:                                                                      | 
            for i in prange(len(out)):---------------------------------------------| #3
                a_i = np.empty(MAX_DIMS, np.int32)  # numpy buffers for indices    | 
                b_i = np.empty(MAX_DIMS, np.int32)                                 | 
                out_i = np.empty(MAX_DIMS, np.int32)                               | 
                # convert the positions to indicees                                | 
                to_index(i, out_shape, out_i)                                      | 
                broadcast_index(out_i, out_shape, a_shape, a_i)                    | 
                broadcast_index(out_i, out_shape, b_shape, b_i)                    | 
                # convert back to positions                                        | 
                a_pos = index_to_position(a_i, a_strides)                          | 
                b_pos = index_to_position(b_i, b_strides)                          | 
                out_pos = index_to_position(out_i, out_strides)                    | 
                out[out_pos] = fn(a_storage[a_pos], b_storage[b_pos])              | 
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 2 parallel for-
loop(s) (originating from loops labelled: #2, #3).
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel structure is already optimal.
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
 
---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
The memory allocation derived from the instruction at 
/content/mod3-JohnnaLiu999/minitorch/fast_ops.py (238) is hoisted out of the 
parallel loop labelled #3 (it will be performed before the loop is executed and 
reused inside the loop):
   Allocation:: a_i = np.empty(MAX_DIMS, np.int32)  # numpy buffers for indices
    - numpy.empty() is used for the allocation.
The memory allocation derived from the instruction at 
/content/mod3-JohnnaLiu999/minitorch/fast_ops.py (239) is hoisted out of the 
parallel loop labelled #3 (it will be performed before the loop is executed and 
reused inside the loop):
   Allocation:: b_i = np.empty(MAX_DIMS, np.int32)
    - numpy.empty() is used for the allocation.
The memory allocation derived from the instruction at 
/content/mod3-JohnnaLiu999/minitorch/fast_ops.py (240) is hoisted out of the 
parallel loop labelled #3 (it will be performed before the loop is executed and 
reused inside the loop):
   Allocation:: out_i = np.empty(MAX_DIMS, np.int32)
    - numpy.empty() is used for the allocation.
None
REDUCE
 
================================================================================
 Parallel Accelerator Optimizing:  Function tensor_reduce.<locals>._reduce, 
/content/mod3-JohnnaLiu999/minitorch/fast_ops.py (275)  
================================================================================


Parallel loop listing for  Function tensor_reduce.<locals>._reduce, /content/mod3-JohnnaLiu999/minitorch/fast_ops.py (275) 
----------------------------------------------------------------------------------------|loop #ID
    def _reduce(                                                                        | 
        out: Storage,                                                                   | 
        out_shape: Shape,                                                               | 
        out_strides: Strides,                                                           | 
        a_storage: Storage,                                                             | 
        a_shape: Shape,                                                                 | 
        a_strides: Strides,                                                             | 
        reduce_dim: int,                                                                | 
    ) -> None:                                                                          | 
        # TODO: Implement for Task 3.1.                                                 | 
        # raise NotImplementedError("Need to implement for Task 3.1")                   | 
        # parallel main loop                                                            | 
        reduce_size = a_shape[reduce_dim]                                               | 
        reduce_stride = a_strides[reduce_dim]                                           | 
        for i in prange(len(out)):------------------------------------------------------| #4
            out_i = np.empty(MAX_DIMS, np.int32)  # numpy buffers                       | 
            to_index(i, out_shape, out_i)  # convert position to index                  | 
            out_pos = index_to_position(out_i, out_strides)                             | 
            in_pos = index_to_position(out_i, a_strides)  # calls outside inner loop    | 
            # current output value                                                      | 
            cur = out[out_pos]                                                          | 
            for _ in range(reduce_size):                                                | 
                cur = fn(cur, a_storage[in_pos])                                        | 
                in_pos += reduce_stride                                                 | 
            out[out_pos] = cur                                                          | 
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 1 parallel for-
loop(s) (originating from loops labelled: #4).
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel structure is already optimal.
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
 
---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
The memory allocation derived from the instruction at 
/content/mod3-JohnnaLiu999/minitorch/fast_ops.py (290) is hoisted out of the 
parallel loop labelled #4 (it will be performed before the loop is executed and 
reused inside the loop):
   Allocation:: out_i = np.empty(MAX_DIMS, np.int32)  # numpy buffers
    - numpy.empty() is used for the allocation.
None
MATRIX MULTIPLY
 
================================================================================
 Parallel Accelerator Optimizing:  Function _tensor_matrix_multiply, 
/content/mod3-JohnnaLiu999/minitorch/fast_ops.py (304)  
================================================================================


Parallel loop listing for  Function _tensor_matrix_multiply, /content/mod3-JohnnaLiu999/minitorch/fast_ops.py (304) 
----------------------------------------------------------------------------------------|loop #ID
def _tensor_matrix_multiply(                                                            | 
    out: Storage,                                                                       | 
    out_shape: Shape,                                                                   | 
    out_strides: Strides,                                                               | 
    a_storage: Storage,                                                                 | 
    a_shape: Shape,                                                                     | 
    a_strides: Strides,                                                                 | 
    b_storage: Storage,                                                                 | 
    b_shape: Shape,                                                                     | 
    b_strides: Strides,                                                                 | 
) -> None:                                                                              | 
    """NUMBA tensor matrix multiply function.                                           | 
                                                                                        | 
    Should work for any tensor shapes that broadcast as long as                         | 
                                                                                        | 
    ```                                                                                 | 
    assert a_shape[-1] == b_shape[-2]                                                   | 
    ```                                                                                 | 
                                                                                        | 
    Optimizations:                                                                      | 
                                                                                        | 
    * Outer loop in parallel                                                            | 
    * No index buffers or function calls                                                | 
    * Inner loop should have no global writes, 1 multiply.                              | 
                                                                                        | 
                                                                                        | 
    Args:                                                                               | 
    ----                                                                                | 
        out (Storage): storage for `out` tensor                                         | 
        out_shape (Shape): shape for `out` tensor                                       | 
        out_strides (Strides): strides for `out` tensor                                 | 
        a_storage (Storage): storage for `a` tensor                                     | 
        a_shape (Shape): shape for `a` tensor                                           | 
        a_strides (Strides): strides for `a` tensor                                     | 
        b_storage (Storage): storage for `b` tensor                                     | 
        b_shape (Shape): shape for `b` tensor                                           | 
        b_strides (Strides): strides for `b` tensor                                     | 
                                                                                        | 
    Returns:                                                                            | 
    -------                                                                             | 
        None : Fills in `out`                                                           | 
                                                                                        | 
    """                                                                                 | 
    # Get dimensions                                                                    | 
    reduced_size = a_shape[2]                                                           | 
    # Get batch stride or 0 if dimension is 1                                           | 
    a_batch_s = a_strides[0] if a_shape[0] > 1 else 0                                   | 
    b_batch_s = b_strides[0] if b_shape[0] > 1 else 0                                   | 
    # Get row and column strides                                                        | 
    a_row_s = a_strides[1]                                                              | 
    a_col_s = a_strides[2]                                                              | 
    b_row_s = b_strides[1]                                                              | 
    b_col_s = b_strides[2]                                                              | 
    # Parallelize outer loop over batches and rows                                      | 
    for batch in prange(out_shape[0]):--------------------------------------------------| #5
        for i in range(out_shape[1]):                                                   | 
            for j in range(out_shape[2]):                                               | 
                # Calculate output position                                             | 
                out_pos = (                                                             | 
                    batch * out_strides[0] + i * out_strides[1] + j * out_strides[2]    | 
                )                                                                       | 
                total = 0.0                                                             | 
                # Calculate starting positions for this batch/row                       | 
                a_in = batch * a_batch_s + i * a_row_s                                  | 
                b_in = batch * b_batch_s + j * b_col_s                                  | 
                # Inner dot product loop                                                | 
                for k in range(reduced_size):                                           | 
                    a_pos = a_in + k * a_col_s                                          | 
                    b_pos = b_in + k * b_row_s                                          | 
                    total += a_storage[a_pos] * b_storage[b_pos]                        | 
                out[out_pos] = total                                                    | 
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 1 parallel for-
loop(s) (originating from loops labelled: #5).
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel structure is already optimal.
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
 
---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
No allocation hoisting found
None
```

# Task 3.3
```
======================================= test session starts ========================================
platform linux -- Python 3.12.7, pytest-8.3.2, pluggy-1.5.0
rootdir: /content/mod3-JohnnaLiu999
configfile: pyproject.toml
plugins: env-1.1.4, hypothesis-6.54.0
collected 279 items / 222 deselected / 57 selected                                                 

tests/test_tensor_general.py .........................................................       [100%]
...
-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
================== 57 passed, 222 deselected, 4309 warnings in 229.07s (0:03:49) ===================
```

# Task 3.4
```
======================================= test session starts ========================================
platform linux -- Python 3.12.7, pytest-8.3.2, pluggy-1.5.0
rootdir: /content/mod3-JohnnaLiu999
configfile: pyproject.toml
plugins: env-1.1.4, hypothesis-6.54.0
collected 279 items / 272 deselected / 7 selected                                                  

tests/test_tensor_general.py .......                                                         [100%]
...
-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
========================= 7 passed, 272 deselected, 141 warnings in 9.82s ==========================
```

![image](https://github.com/user-attachments/assets/821954a6-f174-446f-b079-8c548566f512)

![image](https://github.com/user-attachments/assets/dc70a389-3cc8-4a02-b500-4f57cdeb98b1)


# Task 3.5
Configs: 
- Learning Rate: 0.05 
- Hidden Layers: 100 or 200 (only for split dataset)
- Epochs: 500

'simple' dataset training using GPU
```
Epoch 0 | Loss: 5.765027159619738 | Correct: 35 | Time: 4.34216s/epoch
Epoch 10 | Loss: 0.9534163812190063 | Correct: 47 | Time: 1.94768s/epoch
Epoch 20 | Loss: 2.1256881728072665 | Correct: 49 | Time: 1.98361s/epoch
Epoch 30 | Loss: 0.55952880847853 | Correct: 50 | Time: 2.18798s/epoch
Epoch 40 | Loss: 0.3838986826735673 | Correct: 50 | Time: 1.97126s/epoch
Epoch 50 | Loss: 0.17026185754849837 | Correct: 49 | Time: 2.60281s/epoch
Epoch 60 | Loss: 0.36207419340560665 | Correct: 49 | Time: 2.69904s/epoch
Epoch 70 | Loss: 0.8409530868325024 | Correct: 50 | Time: 1.95353s/epoch
Epoch 80 | Loss: 0.04948289217008585 | Correct: 50 | Time: 2.00253s/epoch
Epoch 90 | Loss: 0.2364764160465918 | Correct: 50 | Time: 2.26350s/epoch
Epoch 100 | Loss: 0.11193687488054439 | Correct: 50 | Time: 2.00322s/epoch
Epoch 110 | Loss: 0.10524398733839649 | Correct: 50 | Time: 1.91669s/epoch
Epoch 120 | Loss: 0.06856915556681838 | Correct: 50 | Time: 2.12849s/epoch
Epoch 130 | Loss: 0.6639801789821168 | Correct: 50 | Time: 1.90563s/epoch
Epoch 140 | Loss: 0.14125455787297042 | Correct: 50 | Time: 1.96990s/epoch
Epoch 150 | Loss: 0.6614307495335242 | Correct: 50 | Time: 1.99831s/epoch
Epoch 160 | Loss: 0.3206912009413979 | Correct: 50 | Time: 2.05744s/epoch
Epoch 170 | Loss: 0.04045269710012697 | Correct: 50 | Time: 1.93698s/epoch
Epoch 180 | Loss: 0.2856128237847032 | Correct: 50 | Time: 1.90143s/epoch
Epoch 190 | Loss: 0.5913458867560847 | Correct: 50 | Time: 1.88931s/epoch
Epoch 200 | Loss: 0.5486196520910821 | Correct: 50 | Time: 1.93105s/epoch
Epoch 210 | Loss: 0.01733237276815438 | Correct: 50 | Time: 1.91534s/epoch
Epoch 220 | Loss: 0.034455863582157166 | Correct: 50 | Time: 1.92666s/epoch
Epoch 230 | Loss: 0.1641533533512292 | Correct: 50 | Time: 2.00780s/epoch
Epoch 240 | Loss: 0.10416086435468994 | Correct: 50 | Time: 1.90791s/epoch
Epoch 250 | Loss: 0.16112091380490035 | Correct: 50 | Time: 2.16476s/epoch
Epoch 260 | Loss: 0.41993949294263294 | Correct: 50 | Time: 1.92472s/epoch
Epoch 270 | Loss: 0.11732638438020893 | Correct: 50 | Time: 1.95586s/epoch
Epoch 280 | Loss: 0.031082865147850712 | Correct: 50 | Time: 2.49982s/epoch
Epoch 290 | Loss: 0.1704472685342568 | Correct: 50 | Time: 1.97572s/epoch
Epoch 300 | Loss: 0.010353512486504216 | Correct: 50 | Time: 1.95913s/epoch
Epoch 310 | Loss: 0.1108939772958178 | Correct: 50 | Time: 2.35853s/epoch
Epoch 320 | Loss: 0.31850867552874185 | Correct: 50 | Time: 1.92972s/epoch
Epoch 330 | Loss: 0.016490039304779144 | Correct: 50 | Time: 1.96462s/epoch
Epoch 340 | Loss: 0.2716912349456285 | Correct: 50 | Time: 1.96955s/epoch
Epoch 350 | Loss: 0.2996942297849032 | Correct: 50 | Time: 1.93588s/epoch
Epoch 360 | Loss: 0.0788098805572624 | Correct: 50 | Time: 2.01492s/epoch
Epoch 370 | Loss: 0.009189095156997827 | Correct: 50 | Time: 1.90452s/epoch
Epoch 380 | Loss: 0.06966740471964418 | Correct: 50 | Time: 2.12885s/epoch
Epoch 390 | Loss: 0.0049760022614939985 | Correct: 50 | Time: 1.92780s/epoch
Epoch 400 | Loss: 0.05628584953923183 | Correct: 50 | Time: 1.99749s/epoch
Epoch 410 | Loss: 0.007634162671345096 | Correct: 50 | Time: 2.62700s/epoch
Epoch 420 | Loss: 0.005098617357250931 | Correct: 50 | Time: 1.96826s/epoch
Epoch 430 | Loss: 0.056855338204177354 | Correct: 50 | Time: 1.91652s/epoch
Epoch 440 | Loss: 0.25053327386687974 | Correct: 50 | Time: 2.40220s/epoch
Epoch 450 | Loss: 0.04827979225204982 | Correct: 50 | Time: 1.91437s/epoch
Epoch 460 | Loss: 0.005359753532768405 | Correct: 50 | Time: 1.92668s/epoch
Epoch 470 | Loss: 0.23822648583887032 | Correct: 50 | Time: 1.94766s/epoch
Epoch 480 | Loss: 0.01680058117070367 | Correct: 50 | Time: 1.99735s/epoch
Epoch 490 | Loss: 0.004772854121969007 | Correct: 50 | Time: 1.96024s/epoch
```

'simple' dataset training using CPU
```
Epoch 0 | Loss: 5.449965934886492 | Correct: 37 | Time: 14.09697s/epoch
Epoch 10 | Loss: 2.9288103061299884 | Correct: 45 | Time: 0.62572s/epoch
Epoch 20 | Loss: 0.7589870436098076 | Correct: 49 | Time: 0.62707s/epoch
Epoch 30 | Loss: 0.46688939615800584 | Correct: 50 | Time: 0.64105s/epoch
Epoch 40 | Loss: 1.3549148924415015 | Correct: 50 | Time: 0.62160s/epoch
Epoch 50 | Loss: 0.41198054565711933 | Correct: 50 | Time: 1.17691s/epoch
Epoch 60 | Loss: 0.3717656629502658 | Correct: 50 | Time: 0.65639s/epoch
Epoch 70 | Loss: 0.3881078363386985 | Correct: 50 | Time: 0.62454s/epoch
Epoch 80 | Loss: 0.3941453279487028 | Correct: 50 | Time: 0.62603s/epoch
Epoch 90 | Loss: 0.11314811509701407 | Correct: 50 | Time: 0.63311s/epoch
Epoch 100 | Loss: 0.38866002566571456 | Correct: 50 | Time: 0.62718s/epoch
Epoch 110 | Loss: 0.471495525337858 | Correct: 50 | Time: 0.62284s/epoch
Epoch 120 | Loss: 0.3422708034948208 | Correct: 50 | Time: 1.19102s/epoch
Epoch 130 | Loss: 0.3375323005636115 | Correct: 50 | Time: 0.65721s/epoch
Epoch 140 | Loss: 0.2701493875564745 | Correct: 50 | Time: 0.66055s/epoch
Epoch 150 | Loss: 0.07095108271187833 | Correct: 50 | Time: 0.64123s/epoch
Epoch 160 | Loss: 0.5992154744187376 | Correct: 50 | Time: 0.63660s/epoch
Epoch 170 | Loss: 0.5102392680076066 | Correct: 50 | Time: 0.63866s/epoch
Epoch 180 | Loss: 0.3970085005999474 | Correct: 50 | Time: 0.83022s/epoch
Epoch 190 | Loss: 0.09814746277925294 | Correct: 50 | Time: 0.62859s/epoch
Epoch 200 | Loss: 0.2801502487965675 | Correct: 50 | Time: 0.62987s/epoch
Epoch 210 | Loss: 0.29084200602760824 | Correct: 50 | Time: 0.62915s/epoch
Epoch 220 | Loss: 0.007589234784062675 | Correct: 50 | Time: 0.62783s/epoch
Epoch 230 | Loss: 0.08245549650095607 | Correct: 50 | Time: 0.64827s/epoch
Epoch 240 | Loss: 0.24987642524340106 | Correct: 50 | Time: 1.11767s/epoch
Epoch 250 | Loss: 0.03714357084019538 | Correct: 50 | Time: 0.63870s/epoch
Epoch 260 | Loss: 0.03909169733912681 | Correct: 50 | Time: 0.63359s/epoch
Epoch 270 | Loss: 0.289775259053993 | Correct: 50 | Time: 0.63485s/epoch
Epoch 280 | Loss: 0.18981115384297073 | Correct: 50 | Time: 0.61919s/epoch
Epoch 290 | Loss: 0.18469717573978195 | Correct: 50 | Time: 0.65742s/epoch
Epoch 300 | Loss: 0.0883351558653421 | Correct: 50 | Time: 0.64676s/epoch
Epoch 310 | Loss: 0.03394705921698518 | Correct: 50 | Time: 0.97190s/epoch
Epoch 320 | Loss: 0.07064855493152486 | Correct: 50 | Time: 0.65807s/epoch
Epoch 330 | Loss: 0.5018735638613961 | Correct: 50 | Time: 0.65365s/epoch
Epoch 340 | Loss: 0.0003929629515040442 | Correct: 50 | Time: 0.61927s/epoch
Epoch 350 | Loss: 0.00010511628135933993 | Correct: 50 | Time: 0.65067s/epoch
Epoch 360 | Loss: 0.013879347899160609 | Correct: 50 | Time: 0.62967s/epoch
Epoch 370 | Loss: 0.4468108071965555 | Correct: 50 | Time: 0.63822s/epoch
Epoch 380 | Loss: 0.3812196680825472 | Correct: 50 | Time: 1.04385s/epoch
Epoch 390 | Loss: 0.0401277870419913 | Correct: 50 | Time: 0.66508s/epoch
Epoch 400 | Loss: 0.0032384135587524127 | Correct: 50 | Time: 0.70167s/epoch
Epoch 410 | Loss: 0.001421597048152832 | Correct: 50 | Time: 0.63578s/epoch
Epoch 420 | Loss: 0.0359160435800076 | Correct: 50 | Time: 0.64574s/epoch
Epoch 430 | Loss: 0.2047143878287738 | Correct: 50 | Time: 0.64224s/epoch
Epoch 440 | Loss: 0.06230453694806781 | Correct: 50 | Time: 0.64585s/epoch
Epoch 450 | Loss: 0.13144124000282004 | Correct: 50 | Time: 1.15362s/epoch
Epoch 460 | Loss: 0.13332454580790565 | Correct: 50 | Time: 0.63945s/epoch
Epoch 470 | Loss: 0.0007830463866001901 | Correct: 50 | Time: 0.64179s/epoch
Epoch 480 | Loss: 0.11773128596495687 | Correct: 50 | Time: 0.66954s/epoch
Epoch 490 | Loss: 0.009626992535771348 | Correct: 50 | Time: 0.63177s/epoch
```

'split' dataset training using GPU
```
Epoch 0 | Loss: 3.655043666362947 | Correct: 37 | Time: 4.05834s/epoch
Epoch 10 | Loss: 5.25560408096986 | Correct: 41 | Time: 1.93139s/epoch
Epoch 20 | Loss: 3.93135812758399 | Correct: 39 | Time: 1.91608s/epoch
Epoch 30 | Loss: 3.223670603634326 | Correct: 38 | Time: 1.95037s/epoch
Epoch 40 | Loss: 3.830785981381479 | Correct: 45 | Time: 1.98077s/epoch
Epoch 50 | Loss: 3.139746932003766 | Correct: 47 | Time: 2.18305s/epoch
Epoch 60 | Loss: 2.4847687080762397 | Correct: 48 | Time: 1.97072s/epoch
Epoch 70 | Loss: 2.8650680507891195 | Correct: 48 | Time: 1.92686s/epoch
Epoch 80 | Loss: 2.6970710597547427 | Correct: 45 | Time: 2.50433s/epoch
Epoch 90 | Loss: 1.0771701051125586 | Correct: 48 | Time: 1.90682s/epoch
Epoch 100 | Loss: 2.255713485864674 | Correct: 50 | Time: 2.19110s/epoch
Epoch 110 | Loss: 2.100454064532166 | Correct: 49 | Time: 2.36736s/epoch
Epoch 120 | Loss: 0.8804868895054903 | Correct: 48 | Time: 1.99358s/epoch
Epoch 130 | Loss: 0.9686110766813946 | Correct: 50 | Time: 1.92405s/epoch
Epoch 140 | Loss: 1.5523049893569751 | Correct: 48 | Time: 2.23251s/epoch
Epoch 150 | Loss: 2.0548039279933503 | Correct: 47 | Time: 1.92400s/epoch
Epoch 160 | Loss: 1.380297934060456 | Correct: 50 | Time: 1.91488s/epoch
Epoch 170 | Loss: 0.4749889773629705 | Correct: 48 | Time: 1.91357s/epoch
Epoch 180 | Loss: 2.165309167468098 | Correct: 47 | Time: 1.94768s/epoch
Epoch 190 | Loss: 1.1948608117480055 | Correct: 47 | Time: 2.11126s/epoch
Epoch 200 | Loss: 0.7738537346837655 | Correct: 50 | Time: 1.90915s/epoch
Epoch 210 | Loss: 3.2523313865473913 | Correct: 45 | Time: 2.22400s/epoch
Epoch 220 | Loss: 1.1521288104142768 | Correct: 50 | Time: 1.92637s/epoch
Epoch 230 | Loss: 0.2927415425026352 | Correct: 50 | Time: 1.98896s/epoch
Epoch 240 | Loss: 1.475633739440481 | Correct: 49 | Time: 2.59962s/epoch
Epoch 250 | Loss: 0.240507006380516 | Correct: 50 | Time: 1.97875s/epoch
Epoch 260 | Loss: 1.2519193250475986 | Correct: 50 | Time: 1.92186s/epoch
Epoch 270 | Loss: 0.7857537110107924 | Correct: 50 | Time: 2.48902s/epoch
Epoch 280 | Loss: 0.3684098117937648 | Correct: 49 | Time: 1.90888s/epoch
Epoch 290 | Loss: 0.6257601384922745 | Correct: 49 | Time: 1.98367s/epoch
Epoch 300 | Loss: 0.8882985124914247 | Correct: 50 | Time: 2.26130s/epoch
Epoch 310 | Loss: 1.5557445703382073 | Correct: 48 | Time: 1.93890s/epoch
Epoch 320 | Loss: 0.5064474021540759 | Correct: 49 | Time: 1.93465s/epoch
Epoch 330 | Loss: 0.14141704179741674 | Correct: 50 | Time: 2.51422s/epoch
Epoch 340 | Loss: 0.6959514802057645 | Correct: 50 | Time: 1.91527s/epoch
Epoch 350 | Loss: 0.2649469600363411 | Correct: 50 | Time: 1.90810s/epoch
Epoch 360 | Loss: 0.4399192172269327 | Correct: 50 | Time: 2.35892s/epoch
Epoch 370 | Loss: 0.48282441994396735 | Correct: 50 | Time: 1.92474s/epoch
Epoch 380 | Loss: 0.4907811264374774 | Correct: 50 | Time: 1.98395s/epoch
Epoch 390 | Loss: 1.1167702705093308 | Correct: 50 | Time: 1.93370s/epoch
Epoch 400 | Loss: 0.200968655083378 | Correct: 50 | Time: 1.98432s/epoch
Epoch 410 | Loss: 0.2758794987756669 | Correct: 50 | Time: 1.90550s/epoch
Epoch 420 | Loss: 0.7524538729854494 | Correct: 50 | Time: 1.98710s/epoch
Epoch 430 | Loss: 0.8686072415030033 | Correct: 50 | Time: 1.90032s/epoch
Epoch 440 | Loss: 0.46322875967190147 | Correct: 50 | Time: 1.93342s/epoch
Epoch 450 | Loss: 0.7431163210089452 | Correct: 50 | Time: 2.08460s/epoch
Epoch 460 | Loss: 0.13350862630052715 | Correct: 50 | Time: 1.90494s/epoch
Epoch 470 | Loss: 0.6356288165550351 | Correct: 50 | Time: 1.88979s/epoch
Epoch 480 | Loss: 0.9041049592251167 | Correct: 48 | Time: 2.03588s/epoch
Epoch 490 | Loss: 0.30247847373512743 | Correct: 50 | Time: 1.94054s/epoch
```

'split' dataset training using CPU
```
Epoch 0 | Loss: 6.674959109151552 | Correct: 36 | Time: 13.98576s/epoch
Epoch 10 | Loss: 4.601279091338706 | Correct: 41 | Time: 0.63864s/epoch
Epoch 20 | Loss: 4.778498973279078 | Correct: 43 | Time: 0.92569s/epoch
Epoch 30 | Loss: 5.6304586998674395 | Correct: 44 | Time: 0.62055s/epoch
Epoch 40 | Loss: 4.097335209863614 | Correct: 45 | Time: 0.62516s/epoch
Epoch 50 | Loss: 3.046951376447207 | Correct: 45 | Time: 0.64526s/epoch
Epoch 60 | Loss: 3.335017167243209 | Correct: 47 | Time: 0.62420s/epoch
Epoch 70 | Loss: 3.196481373992077 | Correct: 49 | Time: 0.61919s/epoch
Epoch 80 | Loss: 2.0359532333528874 | Correct: 49 | Time: 0.61944s/epoch
Epoch 90 | Loss: 1.9997865734239202 | Correct: 48 | Time: 1.13403s/epoch
Epoch 100 | Loss: 2.047102924099448 | Correct: 47 | Time: 0.62441s/epoch
Epoch 110 | Loss: 1.3632873498407534 | Correct: 48 | Time: 0.61846s/epoch
Epoch 120 | Loss: 1.2642470866160078 | Correct: 49 | Time: 0.64025s/epoch
Epoch 130 | Loss: 1.0500818628456052 | Correct: 47 | Time: 0.62776s/epoch
Epoch 140 | Loss: 0.9940474318640523 | Correct: 48 | Time: 0.63421s/epoch
Epoch 150 | Loss: 1.7233152363406692 | Correct: 48 | Time: 0.62059s/epoch
Epoch 160 | Loss: 2.012401711973082 | Correct: 50 | Time: 0.77591s/epoch
Epoch 170 | Loss: 1.0623548002682437 | Correct: 50 | Time: 0.61339s/epoch
Epoch 180 | Loss: 0.6408816044832595 | Correct: 50 | Time: 0.92105s/epoch
Epoch 190 | Loss: 1.1730776866748287 | Correct: 50 | Time: 0.63040s/epoch
Epoch 200 | Loss: 0.8606585227725859 | Correct: 50 | Time: 0.61853s/epoch
Epoch 210 | Loss: 0.7045902297194245 | Correct: 50 | Time: 0.62049s/epoch
Epoch 220 | Loss: 0.887958614945504 | Correct: 50 | Time: 0.62759s/epoch
Epoch 230 | Loss: 1.3776201054184138 | Correct: 50 | Time: 0.63635s/epoch
Epoch 240 | Loss: 0.6126510619784908 | Correct: 50 | Time: 0.61464s/epoch
Epoch 250 | Loss: 0.9629244460670897 | Correct: 50 | Time: 1.16860s/epoch
Epoch 260 | Loss: 1.8950889522635743 | Correct: 48 | Time: 0.63216s/epoch
Epoch 270 | Loss: 0.8425372595808674 | Correct: 50 | Time: 0.63443s/epoch
Epoch 280 | Loss: 0.589010856339415 | Correct: 49 | Time: 0.61675s/epoch
Epoch 290 | Loss: 0.46055631633535166 | Correct: 50 | Time: 0.63234s/epoch
Epoch 300 | Loss: 0.8178350026813588 | Correct: 49 | Time: 0.79326s/epoch
Epoch 310 | Loss: 0.7023039001234718 | Correct: 50 | Time: 0.63284s/epoch
Epoch 320 | Loss: 1.702171257658108 | Correct: 49 | Time: 0.87276s/epoch
Epoch 330 | Loss: 1.2174659773982957 | Correct: 50 | Time: 0.63368s/epoch
Epoch 340 | Loss: 0.49362498771418734 | Correct: 50 | Time: 0.62698s/epoch
Epoch 350 | Loss: 0.8373622947915972 | Correct: 50 | Time: 0.62589s/epoch
Epoch 360 | Loss: 0.8396855892423177 | Correct: 50 | Time: 0.62512s/epoch
Epoch 370 | Loss: 0.4930622036245287 | Correct: 50 | Time: 0.61853s/epoch
Epoch 380 | Loss: 0.12936886739785833 | Correct: 50 | Time: 0.63604s/epoch
Epoch 390 | Loss: 0.609715273768744 | Correct: 50 | Time: 1.14941s/epoch
Epoch 400 | Loss: 0.5011494420637628 | Correct: 50 | Time: 0.62765s/epoch
Epoch 410 | Loss: 0.6208135375200715 | Correct: 50 | Time: 0.62503s/epoch
Epoch 420 | Loss: 1.0339383263381943 | Correct: 50 | Time: 0.63001s/epoch
Epoch 430 | Loss: 0.10723576860854497 | Correct: 50 | Time: 0.62111s/epoch
Epoch 440 | Loss: 0.6180684308974912 | Correct: 50 | Time: 0.61716s/epoch
Epoch 450 | Loss: 0.7176367980582675 | Correct: 50 | Time: 0.63614s/epoch
Epoch 460 | Loss: 0.5862544455143789 | Correct: 50 | Time: 1.08358s/epoch
Epoch 470 | Loss: 0.3629082030049074 | Correct: 50 | Time: 0.62349s/epoch
Epoch 480 | Loss: 0.09296458273209522 | Correct: 50 | Time: 0.66987s/epoch
Epoch 490 | Loss: 0.25075502912582837 | Correct: 50 | Time: 0.62078s/epoch
```

'xor' dataset training using GPU
```
Epoch 0 | Loss: 5.883163505523777 | Correct: 30 | Time: 4.14973s/epoch
Epoch 10 | Loss: 4.595401162933809 | Correct: 41 | Time: 1.96758s/epoch
Epoch 20 | Loss: 4.755474340839483 | Correct: 42 | Time: 1.92153s/epoch
Epoch 30 | Loss: 3.571460929405217 | Correct: 44 | Time: 2.51604s/epoch
Epoch 40 | Loss: 4.647876113543918 | Correct: 44 | Time: 1.97167s/epoch
Epoch 50 | Loss: 4.289918053821859 | Correct: 42 | Time: 1.89872s/epoch
Epoch 60 | Loss: 4.389262726732987 | Correct: 44 | Time: 2.63666s/epoch
Epoch 70 | Loss: 4.682583200564553 | Correct: 44 | Time: 1.93370s/epoch
Epoch 80 | Loss: 3.426111877627391 | Correct: 47 | Time: 1.94755s/epoch
Epoch 90 | Loss: 4.112965807996298 | Correct: 48 | Time: 2.57975s/epoch
Epoch 100 | Loss: 2.30958494129732 | Correct: 48 | Time: 1.98293s/epoch
Epoch 110 | Loss: 1.8260015325568457 | Correct: 45 | Time: 1.89842s/epoch
Epoch 120 | Loss: 3.8105757535894282 | Correct: 48 | Time: 2.74338s/epoch
Epoch 130 | Loss: 1.7709439270603387 | Correct: 45 | Time: 1.90641s/epoch
Epoch 140 | Loss: 2.8414417905190965 | Correct: 48 | Time: 1.95052s/epoch
Epoch 150 | Loss: 1.8418739292236814 | Correct: 48 | Time: 2.32077s/epoch
Epoch 160 | Loss: 2.3977051553221465 | Correct: 48 | Time: 1.91988s/epoch
Epoch 170 | Loss: 0.6882556641996804 | Correct: 47 | Time: 1.89894s/epoch
Epoch 180 | Loss: 1.5687825383675336 | Correct: 47 | Time: 2.45785s/epoch
Epoch 190 | Loss: 1.8880774336929196 | Correct: 48 | Time: 1.88811s/epoch
Epoch 200 | Loss: 0.5861113005512139 | Correct: 49 | Time: 1.90334s/epoch
Epoch 210 | Loss: 1.8486867189672176 | Correct: 48 | Time: 2.47067s/epoch
Epoch 220 | Loss: 1.6190568135930739 | Correct: 46 | Time: 1.91587s/epoch
Epoch 230 | Loss: 0.5412675963604487 | Correct: 47 | Time: 2.00436s/epoch
Epoch 240 | Loss: 2.1930018498427066 | Correct: 49 | Time: 2.60582s/epoch
Epoch 250 | Loss: 2.037724666250185 | Correct: 46 | Time: 1.95624s/epoch
Epoch 260 | Loss: 1.429423241149969 | Correct: 49 | Time: 1.94515s/epoch
Epoch 270 | Loss: 1.5955910649493075 | Correct: 49 | Time: 2.48402s/epoch
Epoch 280 | Loss: 1.3295210806833455 | Correct: 46 | Time: 1.91642s/epoch
Epoch 290 | Loss: 1.055198620384776 | Correct: 49 | Time: 1.95195s/epoch
Epoch 300 | Loss: 0.6913480875442692 | Correct: 49 | Time: 2.34824s/epoch
Epoch 310 | Loss: 1.3054957839595627 | Correct: 49 | Time: 1.91327s/epoch
Epoch 320 | Loss: 0.3622662274474539 | Correct: 49 | Time: 1.89452s/epoch
Epoch 330 | Loss: 0.5867525689832759 | Correct: 50 | Time: 2.29793s/epoch
Epoch 340 | Loss: 1.0218655494041133 | Correct: 49 | Time: 1.89259s/epoch
Epoch 350 | Loss: 0.2944676691640718 | Correct: 49 | Time: 1.89813s/epoch
Epoch 360 | Loss: 2.5305536469716365 | Correct: 48 | Time: 2.35669s/epoch
Epoch 370 | Loss: 0.5883878278565273 | Correct: 49 | Time: 1.90613s/epoch
Epoch 380 | Loss: 0.8798975123046892 | Correct: 49 | Time: 1.94952s/epoch
Epoch 390 | Loss: 0.6099035617764259 | Correct: 50 | Time: 2.47346s/epoch
Epoch 400 | Loss: 1.0781147875706856 | Correct: 49 | Time: 1.96168s/epoch
Epoch 410 | Loss: 0.31691735708206864 | Correct: 49 | Time: 1.89570s/epoch
Epoch 420 | Loss: 0.4376535911759152 | Correct: 50 | Time: 2.50079s/epoch
Epoch 430 | Loss: 0.3916867717370425 | Correct: 49 | Time: 1.88432s/epoch
Epoch 440 | Loss: 0.39987659195928416 | Correct: 49 | Time: 1.90199s/epoch
Epoch 450 | Loss: 1.333368542562669 | Correct: 49 | Time: 2.48123s/epoch
Epoch 460 | Loss: 0.12140349471580167 | Correct: 50 | Time: 1.89085s/epoch
Epoch 470 | Loss: 1.9598113018028465 | Correct: 49 | Time: 1.88413s/epoch
Epoch 480 | Loss: 1.0486918912027314 | Correct: 50 | Time: 2.44209s/epoch
Epoch 490 | Loss: 1.2249429534175813 | Correct: 50 | Time: 2.39906s/epoch
```
'xor' dataset training using CPU
```
Epoch 0 | Loss: 5.974280915790409 | Correct: 34 | Time: 13.68929s/epoch
Epoch 10 | Loss: 5.494134704476132 | Correct: 46 | Time: 0.63459s/epoch
Epoch 20 | Loss: 3.6155501594940027 | Correct: 44 | Time: 0.65285s/epoch
Epoch 30 | Loss: 4.689595287673039 | Correct: 45 | Time: 0.62058s/epoch
Epoch 40 | Loss: 5.367111873991817 | Correct: 44 | Time: 0.63158s/epoch
Epoch 50 | Loss: 4.515010896842426 | Correct: 45 | Time: 0.63434s/epoch
Epoch 60 | Loss: 1.5867021629069582 | Correct: 46 | Time: 0.63546s/epoch
Epoch 70 | Loss: 1.8423800193777784 | Correct: 46 | Time: 1.12811s/epoch
Epoch 80 | Loss: 3.7521232305814745 | Correct: 45 | Time: 0.63893s/epoch
Epoch 90 | Loss: 2.554203106237849 | Correct: 45 | Time: 0.64199s/epoch
Epoch 100 | Loss: 4.1953752694611754 | Correct: 46 | Time: 0.62804s/epoch
Epoch 110 | Loss: 0.5321195309141743 | Correct: 47 | Time: 0.63543s/epoch
Epoch 120 | Loss: 3.6817657488671856 | Correct: 46 | Time: 1.20341s/epoch
Epoch 130 | Loss: 2.816094275395397 | Correct: 47 | Time: 0.63033s/epoch
Epoch 140 | Loss: 2.523245150758128 | Correct: 46 | Time: 0.63890s/epoch
Epoch 150 | Loss: 1.6269430321961837 | Correct: 47 | Time: 0.64461s/epoch
Epoch 160 | Loss: 1.1738731396881799 | Correct: 46 | Time: 0.62166s/epoch
Epoch 170 | Loss: 1.3892860387476864 | Correct: 47 | Time: 0.63401s/epoch
Epoch 180 | Loss: 0.21503780454745686 | Correct: 48 | Time: 0.63595s/epoch
Epoch 190 | Loss: 2.0423183518106587 | Correct: 47 | Time: 1.17878s/epoch
Epoch 200 | Loss: 2.779025355921092 | Correct: 47 | Time: 0.62248s/epoch
Epoch 210 | Loss: 0.22327801569494277 | Correct: 48 | Time: 0.63465s/epoch
Epoch 220 | Loss: 1.1230150487833979 | Correct: 46 | Time: 0.64662s/epoch
Epoch 230 | Loss: 1.1656814165870788 | Correct: 49 | Time: 0.62802s/epoch
Epoch 240 | Loss: 2.041598630519329 | Correct: 48 | Time: 0.63489s/epoch
Epoch 250 | Loss: 1.0036754605117586 | Correct: 48 | Time: 0.63360s/epoch
Epoch 260 | Loss: 0.31016330259404534 | Correct: 47 | Time: 1.18890s/epoch
Epoch 270 | Loss: 0.3289932436450697 | Correct: 49 | Time: 0.62841s/epoch
Epoch 280 | Loss: 0.1330168157663402 | Correct: 49 | Time: 0.63516s/epoch
Epoch 290 | Loss: 1.0537637174972865 | Correct: 49 | Time: 0.63923s/epoch
Epoch 300 | Loss: 0.9941665931610058 | Correct: 49 | Time: 0.62512s/epoch
Epoch 310 | Loss: 1.8229619939284776 | Correct: 49 | Time: 0.63207s/epoch
Epoch 320 | Loss: 2.1488122008692416 | Correct: 49 | Time: 0.63368s/epoch
Epoch 330 | Loss: 1.6631018305193184 | Correct: 49 | Time: 1.03656s/epoch
Epoch 340 | Loss: 0.7539361554021552 | Correct: 50 | Time: 0.63814s/epoch
Epoch 350 | Loss: 0.7392371680914703 | Correct: 49 | Time: 0.63894s/epoch
Epoch 360 | Loss: 0.2757640814660698 | Correct: 50 | Time: 0.62525s/epoch
Epoch 370 | Loss: 0.9376923590705452 | Correct: 47 | Time: 0.65430s/epoch
Epoch 380 | Loss: 0.9648763219552732 | Correct: 50 | Time: 0.64897s/epoch
Epoch 390 | Loss: 0.5911362989210186 | Correct: 50 | Time: 0.62969s/epoch
Epoch 400 | Loss: 0.39285787655229826 | Correct: 50 | Time: 1.07306s/epoch
Epoch 410 | Loss: 0.6660555432997186 | Correct: 48 | Time: 0.63356s/epoch
Epoch 420 | Loss: 2.046288067477715 | Correct: 49 | Time: 1.20698s/epoch
Epoch 430 | Loss: 0.48231915527454744 | Correct: 50 | Time: 0.62464s/epoch
Epoch 440 | Loss: 0.0663760393272809 | Correct: 49 | Time: 0.64651s/epoch
Epoch 450 | Loss: 0.5392097255562077 | Correct: 50 | Time: 0.63753s/epoch
Epoch 460 | Loss: 1.0973769913295235 | Correct: 50 | Time: 0.63672s/epoch
Epoch 470 | Loss: 0.8927002354276323 | Correct: 49 | Time: 0.91649s/epoch
Epoch 480 | Loss: 0.7777393221576349 | Correct: 50 | Time: 0.63341s/epoch
Epoch 490 | Loss: 0.507149233280947 | Correct: 50 | Time: 0.64790s/epoch
```

'split' dataset training using GPU - larger model (200 hidden layers)
```
Epoch 0 | Loss: 5.991442586211891 | Correct: 30 | Time: 15.59952s/epoch
Epoch 10 | Loss: 4.590323917295452 | Correct: 46 | Time: 3.29304s/epoch
Epoch 20 | Loss: 1.7182513382193356 | Correct: 49 | Time: 3.30488s/epoch
Epoch 30 | Loss: 1.5052879301441728 | Correct: 50 | Time: 2.86156s/epoch
Epoch 40 | Loss: 1.1126347509617607 | Correct: 50 | Time: 3.18216s/epoch
Epoch 50 | Loss: 1.9286122885162418 | Correct: 47 | Time: 3.24557s/epoch
Epoch 60 | Loss: 1.1285532251028456 | Correct: 50 | Time: 3.25911s/epoch
Epoch 70 | Loss: 0.9658435613917403 | Correct: 48 | Time: 3.21664s/epoch
Epoch 80 | Loss: 0.6794575371326674 | Correct: 50 | Time: 3.06213s/epoch
Epoch 90 | Loss: 0.6627383032027206 | Correct: 50 | Time: 2.85914s/epoch
Epoch 100 | Loss: 0.7228026644186606 | Correct: 50 | Time: 2.47236s/epoch
Epoch 110 | Loss: 0.8361003480729399 | Correct: 50 | Time: 2.38468s/epoch
Epoch 120 | Loss: 0.2515330183231207 | Correct: 50 | Time: 2.25566s/epoch
Epoch 130 | Loss: 0.1893120519596586 | Correct: 50 | Time: 2.98587s/epoch
Epoch 140 | Loss: 0.48424854632599956 | Correct: 50 | Time: 2.69448s/epoch
Epoch 150 | Loss: 0.4110624586088326 | Correct: 50 | Time: 2.41680s/epoch
Epoch 160 | Loss: 0.3953067301257312 | Correct: 50 | Time: 2.25910s/epoch
Epoch 170 | Loss: 0.2782939775819783 | Correct: 50 | Time: 2.20400s/epoch
Epoch 180 | Loss: 0.8011321543850263 | Correct: 50 | Time: 2.19972s/epoch
Epoch 190 | Loss: 0.2707075885676626 | Correct: 50 | Time: 2.21691s/epoch
Epoch 200 | Loss: 0.4467257960690053 | Correct: 50 | Time: 2.25522s/epoch
Epoch 210 | Loss: 0.18981593085988022 | Correct: 50 | Time: 2.24831s/epoch
Epoch 220 | Loss: 0.3305793590486173 | Correct: 50 | Time: 2.20907s/epoch
Epoch 230 | Loss: 0.2888509182490176 | Correct: 50 | Time: 2.19455s/epoch
Epoch 240 | Loss: 0.3303581997582604 | Correct: 50 | Time: 2.22188s/epoch
Epoch 250 | Loss: 0.08146653771416594 | Correct: 50 | Time: 2.21943s/epoch
Epoch 260 | Loss: 0.024961061836161758 | Correct: 50 | Time: 2.20748s/epoch
Epoch 270 | Loss: 0.17868901236353885 | Correct: 50 | Time: 2.25740s/epoch
Epoch 280 | Loss: 0.15318393747611134 | Correct: 50 | Time: 2.20749s/epoch
Epoch 290 | Loss: 0.07478164696777335 | Correct: 50 | Time: 2.21850s/epoch
Epoch 300 | Loss: 0.044925451981794304 | Correct: 50 | Time: 2.20898s/epoch
Epoch 310 | Loss: 0.2665656396557774 | Correct: 50 | Time: 2.24027s/epoch
Epoch 320 | Loss: 0.23891200604611892 | Correct: 50 | Time: 2.25453s/epoch
Epoch 330 | Loss: 0.1401099071714847 | Correct: 50 | Time: 2.24987s/epoch
Epoch 340 | Loss: 0.32180107188807694 | Correct: 50 | Time: 2.21565s/epoch
Epoch 350 | Loss: 0.018112634797412607 | Correct: 50 | Time: 2.22400s/epoch
Epoch 360 | Loss: 0.037725655540313686 | Correct: 50 | Time: 2.21477s/epoch
Epoch 370 | Loss: 0.020520998937945487 | Correct: 50 | Time: 2.22072s/epoch
Epoch 380 | Loss: 0.12425436515150093 | Correct: 50 | Time: 2.22489s/epoch
Epoch 390 | Loss: 0.09840820591346934 | Correct: 50 | Time: 2.20438s/epoch
Epoch 400 | Loss: 0.11244916468384225 | Correct: 50 | Time: 2.23694s/epoch
Epoch 410 | Loss: 0.16964899939067468 | Correct: 50 | Time: 2.21869s/epoch
Epoch 420 | Loss: 0.03658332943288695 | Correct: 50 | Time: 2.24685s/epoch
Epoch 430 | Loss: 0.158465966752774 | Correct: 50 | Time: 2.22975s/epoch
Epoch 440 | Loss: 0.1304343873997476 | Correct: 50 | Time: 2.28109s/epoch
Epoch 450 | Loss: 0.17844774407794778 | Correct: 50 | Time: 2.21715s/epoch
Epoch 460 | Loss: 0.07266171968630722 | Correct: 50 | Time: 2.21727s/epoch
Epoch 470 | Loss: 0.007862330260699527 | Correct: 50 | Time: 2.20103s/epoch
Epoch 480 | Loss: 0.038440065327844324 | Correct: 50 | Time: 2.19486s/epoch
Epoch 490 | Loss: 0.04611319037125129 | Correct: 50 | Time: 2.21672s/epoch
```
'split' dataset training using CPU - larger model (200 hidden layers)
```
Epoch 0 | Loss: 7.475955470875842 | Correct: 36 | Time: 4.80967s/epoch
Epoch 10 | Loss: 4.1959832601511975 | Correct: 44 | Time: 3.51454s/epoch
Epoch 20 | Loss: 3.0507131140084662 | Correct: 43 | Time: 2.77529s/epoch
Epoch 30 | Loss: 2.9202582975890885 | Correct: 46 | Time: 2.97833s/epoch
Epoch 40 | Loss: 3.714937874983553 | Correct: 47 | Time: 2.88679s/epoch
Epoch 50 | Loss: 1.86460627985268 | Correct: 47 | Time: 2.80937s/epoch
Epoch 60 | Loss: 0.6083720623983744 | Correct: 47 | Time: 2.84589s/epoch
Epoch 70 | Loss: 3.323476415288752 | Correct: 47 | Time: 2.78163s/epoch
Epoch 80 | Loss: 1.773194072337866 | Correct: 48 | Time: 3.47845s/epoch
Epoch 90 | Loss: 2.244568744682269 | Correct: 49 | Time: 3.52690s/epoch
Epoch 100 | Loss: 2.20213815758673 | Correct: 47 | Time: 3.57690s/epoch
Epoch 110 | Loss: 1.1759444720434362 | Correct: 48 | Time: 2.78321s/epoch
Epoch 120 | Loss: 0.9606456380629742 | Correct: 50 | Time: 3.39832s/epoch
Epoch 130 | Loss: 0.9370247154966488 | Correct: 49 | Time: 2.78892s/epoch
Epoch 140 | Loss: 0.6509820218328144 | Correct: 49 | Time: 2.87437s/epoch
Epoch 150 | Loss: 0.3312176231787696 | Correct: 49 | Time: 2.79769s/epoch
Epoch 160 | Loss: 1.5695309876051442 | Correct: 48 | Time: 2.79503s/epoch
Epoch 170 | Loss: 2.0647317348038006 | Correct: 49 | Time: 3.15056s/epoch
Epoch 180 | Loss: 1.0760156746379055 | Correct: 49 | Time: 2.79919s/epoch
Epoch 190 | Loss: 1.7499667054611854 | Correct: 49 | Time: 3.52246s/epoch
Epoch 200 | Loss: 1.4954412464347937 | Correct: 49 | Time: 2.79218s/epoch
Epoch 210 | Loss: 0.3319783785301569 | Correct: 50 | Time: 2.95838s/epoch
Epoch 220 | Loss: 1.0247586618705984 | Correct: 50 | Time: 2.78508s/epoch
Epoch 230 | Loss: 0.7921144217215378 | Correct: 49 | Time: 2.82282s/epoch
Epoch 240 | Loss: 0.8864315209745143 | Correct: 49 | Time: 3.08821s/epoch
Epoch 250 | Loss: 0.2018241763361321 | Correct: 49 | Time: 2.83136s/epoch
Epoch 260 | Loss: 0.7966328788563357 | Correct: 50 | Time: 3.43266s/epoch
Epoch 270 | Loss: 0.1438374337268941 | Correct: 49 | Time: 2.83587s/epoch
Epoch 280 | Loss: 0.3333013379582904 | Correct: 50 | Time: 3.36921s/epoch
Epoch 290 | Loss: 0.22520283029609328 | Correct: 50 | Time: 2.83366s/epoch
Epoch 300 | Loss: 0.6571240609787047 | Correct: 50 | Time: 2.78942s/epoch
Epoch 310 | Loss: 1.5212086148483177 | Correct: 50 | Time: 2.76698s/epoch
Epoch 320 | Loss: 0.5430240681164042 | Correct: 49 | Time: 2.76974s/epoch
Epoch 330 | Loss: 0.1776797317885204 | Correct: 49 | Time: 3.42600s/epoch
Epoch 340 | Loss: 0.042711679493550744 | Correct: 49 | Time: 2.78598s/epoch
Epoch 350 | Loss: 0.9009569370093068 | Correct: 50 | Time: 3.54909s/epoch
Epoch 360 | Loss: 0.2497307210034304 | Correct: 50 | Time: 2.83985s/epoch
Epoch 370 | Loss: 0.07401617263819667 | Correct: 50 | Time: 2.86394s/epoch
Epoch 380 | Loss: 1.1830051706321645 | Correct: 50 | Time: 2.82545s/epoch
Epoch 390 | Loss: 0.2995150664720111 | Correct: 49 | Time: 2.78746s/epoch
Epoch 400 | Loss: 0.17576670603205374 | Correct: 50 | Time: 3.28843s/epoch
Epoch 410 | Loss: 0.15699730097558867 | Correct: 50 | Time: 2.76628s/epoch
Epoch 420 | Loss: 1.0854535409688173 | Correct: 49 | Time: 3.56369s/epoch
Epoch 430 | Loss: 0.2554946380578697 | Correct: 50 | Time: 2.98309s/epoch
Epoch 440 | Loss: 0.189504397469848 | Correct: 50 | Time: 3.02438s/epoch
Epoch 450 | Loss: 0.20698341974059603 | Correct: 50 | Time: 2.80337s/epoch
Epoch 460 | Loss: 0.9399182272303671 | Correct: 50 | Time: 2.76927s/epoch
Epoch 470 | Loss: 0.7744202795236287 | Correct: 50 | Time: 3.10767s/epoch
Epoch 480 | Loss: 0.13332829344287456 | Correct: 50 | Time: 2.77342s/epoch
Epoch 490 | Loss: 0.9421507746495068 | Correct: 50 | Time: 3.51554s/epoch
```

