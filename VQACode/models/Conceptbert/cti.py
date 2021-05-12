import tensorflow as tf

class CTI(tf.keras.layers.Layer):
    def __init__(self, rank, dv, dq, dg, dz, rate=0.1):
        super(CTI, self).__init__()

        self.triAtt = TriAttention(rank, dv, dq, dg)
        self.wzv = tf.keras.layers.Dense(dz,use_bias=False) 
        self.wzq = tf.keras.layers.Dense(dz,use_bias=False) 
        self.wzg = tf.keras.layers.Dense(dz,use_bias=False) 

    def call(self, v, q, g, training=True):
        m = self.triAtt(v, q, g) # (b, nv, nq, ng)
        # print(m.shape)
        # print(v.shape)
        # print(q.shape)
        # print(g.shape)
        # print('start')

        # for i in range(v.shape[1]):
        #     for j in range(q.shape[1]):
        #         for k in range(g.shape[1]):
        #             # (b, dv) * (dv, dz) = (b, dz)
        #             vi = self.wzv(v[:,i,:])
        #             qj = self.wzq(q[:,j,:])
        #             gk = self.wzg(g[:,k,:])
        #             # (b, dz) * (b, dz) * (b, dz) = (b, dz)
        #             hp = tf.math.multiply(vi, qj)
        #             hp = tf.math.multiply(hp, gk)
        #             # (b, dz) = (b, 1) * (b, dz)
        #             z = z + tf.reshape(m[:,i,j,k], [hp.shape[0], 1]) * hp

        # (b, nv, dv) * (dv, dz) = (b, nv, dz)
        # (b, nv, dz) -> (b, dz, nv)
        v = tf.transpose(self.wzv(v), perm=[0, 2, 1])
        q = tf.transpose(self.wzq(q), perm=[0, 2, 1])
        g = tf.transpose(self.wzg(g), perm=[0, 2, 1])
        # (b, dz, nq) -> (b, dz, nq, 1)
        q = tf.expand_dims(q, axis=3)
        g = tf.expand_dims(g, axis=3)
        # (b, dz, nv), (b, nv, nq, ng), (b, dz, nq, 1), (b, dz, ng, 1)
        # -> (b, dz, 1, 1)
        z = tf.einsum('bdv,bvqg,bdqi,bdgj->bdij', v, m, q, g)
        # -> (b, dz)
        z = tf.squeeze(z)
        # print(z.shape)
        return z  # (batch_size, dz)

class TriAttention(tf.keras.layers.Layer):
    def __init__(self, rank, dv, dq, dg, rate=0.1):
        super(TriAttention, self).__init__()

        self.rank = rank
        self.dv = dv
        self.dq = dq
        self.dg = dg
        self.dvr = int(dv/rank)
        self.dqr = int(dq/rank)
        self.dgr = int(dg/rank)

        self.ga = tf.Variable(tf.random.normal([rank, self.dvr, self.dqr, self.dgr], stddev=0.01),
                      name="g")

        self.wv = [tf.keras.layers.Dense(self.dvr,use_bias=False) 
                    for _ in range(rank)]
        self.wq = [tf.keras.layers.Dense(self.dqr,use_bias=False) 
                    for _ in range(rank)]
        self.wg = [tf.keras.layers.Dense(self.dgr,use_bias=False) 
                    for _ in range(rank)]


        # self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, v, q, g, training=True):
        m = 0
        for r in range(self.rank):
            # (nv, dv) * (dv, dvr) = (nv, dvr)
            # v = tf.matmul(v, self.wv[r])
            # q = tf.matmul(q, self.wq[r])
            # g = tf.matmul(g, self.wg[r])
            v = self.wv[r](v)
            q = self.wq[r](q)
            g = self.wg[r](g)
            # # (dvr, dqr, dgr) X1 (b, nv, dvr) = (b, nv, dqr, dgr)
            # temp = mode_product_1(self.ga[r], v)
            # # (b, nv, dqr, dgr) X2 (b, nq, dqr) = (b, nv, nq, dgr)
            # temp = mode_product_2(temp, q)
            # # (b, nv, nq, dgr) X2 (b, ng, dgr) = (b, nv, nq, ng)
            # temp = mode_product_3(temp, g)

            temp = tf.einsum('VQG,BvV,BqQ,BgG->Bvqg', self.ga[r], v, q, g)

            m = m + temp
        return m  # (b, nv, nq, ng)

def mode_product_1(x, u):
    exp = f'VQG,BNV->BNQG'
    return tf.einsum(exp, x, u)

def mode_product_2(x, u):
    exp = f'BVQG,BNQ->BVNG'
    return tf.einsum(exp, x, u)

def mode_product_3(x, u):
    exp = f'BVQG,BNG->BVQN'
    return tf.einsum(exp, x, u)

def scalar_mul(x, u):
    exp = f'B,BD->BD'
    return tf.einsum(exp, x, u)

# def n_mode_product(x, u, n):
#     n = int(n)
#     # We need one letter per dimension
#     # (maybe you could find a workaround for this limitation)
#     if n > 26:
#         raise ValueError('n is too large.')
#     ind = ''.join(chr(ord('a') + i) for i in range(n))
#     exp = f'{ind}K...,BJK->B{ind}J...'
#     return tf.einsum(exp, x, u)

