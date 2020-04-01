import torch
import math

degs = [1, 2, 4, 8, 12, 18]

thetas_dict = {"single": [1.192092800768788e-07,  # m_vals = 1
                          5.978858893805233e-04,  # m_vals = 2
                         #1.123386473528671e-02,
                          5.116619363445086e-02,  # m_vals = 4
                         #1.308487164599470e-01,
                         #2.495289322846698e-01,
                         #4.014582423510481e-01,
                          5.800524627688768e-01,  # m_vals = 8
                         #7.795113374358031e-01,
                         #9.951840790004457e-01,
                         #1.223479542424143e+00,
                          1.461661507209034e+00,  # m_vals = 12
                         #1.707648529608701e+00,
                         #1.959850585959898e+00,
                         #2.217044394974720e+00,
                         #2.478280877521971e+00,
                         #2.742817112698780e+00,
                          3.010066362817634e+00], # m_vals = 18
               "double": [
                          2.220446049250313e-16,  # m_vals = 1
                          2.580956802971767e-08,  # m_vals = 2
                         #1.386347866119121e-05,
                          3.397168839976962e-04,  # m_vals = 4
                         #2.400876357887274e-03,
                         #9.065656407595102e-03,
                         #2.384455532500274e-02,
                          4.991228871115323e-02,  # m_vals = 8
                         #8.957760203223343e-02,
                         #1.441829761614378e-01,
                         #2.142358068451711e-01,
                          2.996158913811580e-01,  # m_vals = 12
                         #3.997775336316795e-01,
                         #5.139146936124294e-01,
                         #6.410835233041199e-01,
                         #7.802874256626574e-01,
                         #9.305328460786568e-01,
                          1.090863719290036e+00]  # m_vals = 18
               }

def matrix_power_two_batch(A, k):
    orig_size = A.size()
    A, k = A.flatten(0, -3), k.flatten()
    ksorted, idx = torch.sort(k)
    # Abusing bincount...
    count = torch.bincount(ksorted)
    nonzero = torch.nonzero(count, as_tuple=False)
    A = torch.matrix_power(A, 2**ksorted[0])
    last = ksorted[0]
    processed = count[nonzero[0]]
    for exp in nonzero[1:]:
        new, last = exp - last, exp
        A[idx[processed:]] = torch.matrix_power(A[idx[processed:]], 2**new.item())
        processed += count[exp]
    return A.reshape(orig_size)


def expm_taylor(A):
    if A.ndimension() < 2 or A.size(-2) != A.size(-1):
        raise ValueError('Expected a square matrix or a batch of squared matrices')

    if A.ndimension() == 2:
        # Just one matrix

        # Trivial case
        if A.size() == (1, 1):
            return torch.exp(A)

        if A.element_size() > 4:
            thetas = thetas_dict["double"]
        else:
            thetas = thetas_dict["single"]

        # No scale-square needed
        # This could be done marginally faster if iterated in reverse
        normA = torch.norm(A, 1).item()
        for deg, theta in zip(degs, thetas):
            if normA <= theta:
                return taylor_approx(A, deg)

        # Scale square
        s = int(math.ceil(math.log2(normA) - math.log2(thetas[-1])))
        A = A * (2**-s)
        X = taylor_approx(A, degs[-1])
        return torch.matrix_power(X, 2**s)
    else:
        # Batching

        # Trivial case
        if A.size()[-2:] == (1, 1):
            return torch.exp(A)

        if A.element_size() > 4:
            thetas = thetas_dict["double"]
        else:
            thetas = thetas_dict["single"]

        normA = torch.norm(A, dim=(-2, -1))

        # Handle trivial case
        if (normA == 0.).all():
            I = torch.eye(A.size(-2), A.size(-1), dtype=A.dtype, device=A.device)
            I = I.expand_as(A)
            return I

        # Handle small normA
        more = normA > thetas[-1]
        k = normA.new_zeros(normA.size(), dtype=torch.long)
        k[more] = torch.ceil(torch.log2(normA[more]) - math.log2(thetas[-1])).long()

        # A = A * 2**(-s)
        A = torch.pow(.5, k.float()).unsqueeze_(-1).unsqueeze_(-1).expand_as(A) * A
        X = taylor_approx(A, degs[-1])
        return matrix_power_two_batch(X, k)


def taylor_approx(A, deg):
    batched = A.ndimension() > 2
    I = torch.eye(A.size(-2), A.size(-1), dtype=A.dtype, device=A.device)
    if batched:
        I = I.expand_as(A)

    if deg >= 2:
        A2 = A @ A
    if deg > 8:
        A3 = A @ A2
    if deg == 18:
        A6 = A3 @ A3

    if deg == 1:
        return I + A
    elif deg == 2:
        return I + A + .5*A2
    elif deg == 4:
        return I + A + A2 @ (.5*I + A/6. + A2/24.)
    elif deg == 8:
        # Minor: Precompute
        SQRT = math.sqrt(177.)
        x3 = 2./3.
        a1 = (1.+SQRT)*x3
        x1 = a1/88.
        x2 = a1/352.
        c0 = (-271.+29.*SQRT)/(315.*x3)
        c1 = (11.*(-1.+SQRT))/(1260.*x3)
        c2 = (11.*(-9.+SQRT))/(5040.*x3)
        c4 = (89.-SQRT)/(5040.*x3*x3)
        y2 = ((857.-58.*SQRT))/630.
        # Matrix products
        A4 = A2 @ (x1*A + x2*A2)
        A8 = (x3*A2 + A4) @ (c0*I + c1*A + c2*A2 + c4*A4)
        return I + A + y2*A2 + A8
    elif deg == 12:
        b = torch.tensor(
		[[-1.86023205146205530824e-02,
		  -5.00702322573317714499e-03,
		  -5.73420122960522249400e-01,
		  -1.33399693943892061476e-01],
		  [ 4.6,
		    9.92875103538486847299e-01,
		   -1.32445561052799642976e-01,
		    1.72990000000000000000e-03],
		  [ 2.11693118299809440730e-01,
		    1.58224384715726723583e-01,
		    1.65635169436727403003e-01,
		    1.07862779315792429308e-02],
		  [ 0.,
		   -1.31810610138301836924e-01,
		   -2.02785554058925905629e-02,
		   -6.75951846863086323186e-03]],
        dtype=A.dtype, device=A.device)

        # We implement the following allowing for batches
        #q31 = a01*I+a11*A+a21*A2+a31*A3
        #q32 = a02*I+a12*A+a22*A2+a32*A3
        #q33 = a03*I+a13*A+a23*A2+a33*A3
        #q34 = a04*I+a14*A+a24*A2+a34*A3
        # Matrix products
        #q61 = q33 + q34 @ q34
        #return (q31 + (q32 + q61) @ q61)

        # Example of non-batched version for reference
        #q = torch.stack([I, A, A2, A3]).repeat(4, 1, 1, 1)
        #b = b.unsqueeze(-1).unsqueeze(-1).expand_as(q)
        #q = (b * q).sum(dim=1)
        #qaux = q[2] + q[3] @ q[3]
        #return q[0] + (q[1] + qaux) @ qaux

        q = torch.stack([I, A, A2, A3], dim=-3).unsqueeze_(-4)
        len_batch = A.ndimension() - 2
        # Expand first dimension
        q_size =  [-1 for _ in range(len_batch)] + [4, -1, -1, -1]
        q = q.expand(*q_size)
        b = b.unsqueeze_(-1).unsqueeze_(-1).expand_as(q)
        q = (b * q).sum(dim=-3)
        if batched:
            # Indexing the third to last dimension, because otherwise we
            # would have to prepend as many 1's as the batch shape for the
            # previous expand_as to work
            qaux = q[..., 2,:,:] + q[..., 3,:,:] @ q[..., 3,:,:]
            return q[..., 0,:,:] + (q[..., 1,:,:] + qaux) @ qaux
        else:
            qaux = q[2] + q[3] @ q[3]
            return q[0] + (q[1] + qaux) @ qaux

    elif deg == 18:
        b = torch.tensor(
	[[0.,
	 -1.00365581030144618291e-01,
         -8.02924648241156932449e-03,
	 -8.92138498045729985177e-04,
          0.],
        [ 0.,
	  3.97849749499645077844e-01,
          1.36783778460411720168e+00,
	  4.98289622525382669416e-01,
         -6.37898194594723280150e-04],
        [-1.09676396052962061844e+01,
	  1.68015813878906206114e+00,
          5.71779846478865511061e-02,
	 -6.98210122488052056106e-03,
          3.34975017086070470649e-05],
        [-9.04316832390810593223e-02,
	 -6.76404519071381882256e-02,
          6.75961301770459654925e-02,
	  2.95552570429315521194e-02,
         -1.39180257516060693404e-05],
        [ 0.,
	  0.,
         -9.23364619367118555360e-02,
	 -1.69364939002081722752e-02,
         -1.40086798182036094347e-05]],
        dtype=A.dtype, device=A.device)

        # We implement the following allowing for batches
        #q31 = a01*I + a11*A + a21*A2 + a31*A3
        #q61 = b01*I + b11*A + b21*A2 + b31*A3 + b61*A6
        #q62 = b02*I + b12*A + b22*A2 + b32*A3 + b62*A6
        #q63 = b03*I + b13*A + b23*A2 + b33*A3 + b63*A6
        #q64 = b04*I + b14*A + b24*A2 + b34*A3 + b64*A6
        #q91 = q31 @ q64 + q63
        #return q61 + (q62 + q91) @ q91
        q = torch.stack([I, A, A2, A3, A6], dim=-3).unsqueeze_(-4)
        len_batch = A.ndimension() - 2
        q_size =  [-1 for _ in range(len_batch)] + [5, -1, -1, -1]
        q = q.expand(*q_size)
        b = b.unsqueeze_(-1).unsqueeze_(-1).expand_as(q)
        q = (b * q).sum(dim=-3)
        if batched:
            # Indexing the third to last dimension, because otherwise we
            # would have to prepend as many 1's as the batch shape for the
            # previous expand_as to work
            qaux = q[..., 0,:,:] @ q[..., 4,:,:] + q[..., 3,:,:]
            return q[..., 1,:,:] + (q[..., 2,:,:] + qaux) @ qaux
        else:
            qaux = q[0] @ q[4] + q[3]
            return q[1] + (q[2] + qaux) @ qaux


def differential(A, E, f):
    n = A.size(-1)
    size_M = list(A.size()[:-2]) + [2*n, 2*n]
    M = A.new_zeros(size_M)
    M[..., :n, :n] = A
    M[..., n:, n:] = A
    M[..., :n, n:] = E
    return f(M)[..., :n, n:]


class expm_taylor_class(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A):
        ctx.save_for_backward(A)
        return expm_taylor(A)

    @staticmethod
    def backward(ctx, G):
        (A,) = ctx.saved_tensors
        # Handle tipical case separately as (dexp)_0 = Id
        if (A == 0).all():
            return G
        else:
            return differential(A.transpose(-2, -1), G, expm_taylor)


expm = expm_taylor_class.apply
