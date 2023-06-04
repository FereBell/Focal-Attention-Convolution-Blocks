from tensorflow.keras.layers import Conv2D, BatchNormalization, Conv2DTranspose, MaxPooling2D, UpSampling2D, Concatenate, Dropout, Activation,  Add, Multiply, ZeroPadding2D
from Tranning import Entre_editado
import tensorflow as tf

def CBAM(entrada, num_kernel, ratio):
  reduccion= int(num_kernel//ratio)
  favr= tf.reduce_mean(entrada, axis= [-1], keepdims= True)
  fmax= tf.reduce_max(entrada, axis= [-1], keepdims= True)
  davr_r= tf.keras.layers.Dense(reduccion, activation= 'relu')(favr)
  dmax_r= tf.keras.layers.Dense(reduccion, activation= 'relu')(fmax)
  davr_a= tf.keras.layers.Dense(num_kernel, activation= 'relu')(davr_r)
  dmax_a= tf.keras.layers.Dense(num_kernel, activation= 'relu')(dmax_r)
  x= tf.add(davr_a, dmax_a)
  x= tf.keras.layers.Activation('sigmoid')(x)
  r= tf.math.multiply(entrada, x)
  favr_s= tf.reduce_mean(r, axis= [-1], keepdims= True)
  fmax_s= tf.reduce_max(r, axis= [-1], keepdims= True)
  c= tf.concat([favr_s, fmax_s], axis= -1)
  c= Conv2D(1, 7, padding= 'same')(c)
  c= BatchNormalization()(c)
  c= tf.keras.layers.Activation('sigmoid')(c)
  salida= tf.math.multiply(x, c)
  return salida

def sub_ventana(entrada, sw, sr, dim):
  mp= (dim//2) + (sr//2)
  mb= (dim//2) - (sr//2)
  sub_w= entrada[:, mb:mp, mb:mp, :]
  sub_w= MaxPooling2D(sw)(sub_w)
  zeros_w= ZeroPadding2D((dim-(sr//sw))//2)(sub_w)

  return Add()([entrada, zeros_w])

def second_p(entrada, sw, sr, dim, n_dims):
  i= 0
  for w, r in zip(sw, sr):
    if i== 0:
      nuevo= sub_ventana(entrada, w, r, dim)
      i= 1
    else:
      nuevo= sub_ventana(nuevo, w, r, dim)

  x= CBAM(nuevo, n_dims, 16)
  y= MHSA(n_dims, nuevo)
  z= Multiply()([x, y])

  return Add()([z, entrada])

def MHSA(n_dims, entrada, heads= 4):
  canal= n_dims//2
  depth= canal// heads
  b_z= tf.shape(entrada)[0]

  q= Conv2D(canal, 1, padding= 'same')(entrada)
  k= Conv2D(canal, 1, padding= 'same')(entrada)
  v= Conv2D(canal, 1, padding= 'same')(entrada)

  q_s= split_heads(q, b_z, heads, depth)
  k_s= split_heads(k, b_z, heads, depth)
  v_s= split_heads(v, b_z, heads, depth)

  scaled_attention= scaled_dot_product_attention(q_s, k_s, v_s)
  scaled_attention= tf.transpose(scaled_attention, perm=[0, 2, 1, 3])
  out= tf.reshape(scaled_attention, [b_z, tf.shape(entrada)[1], tf.shape(entrada)[2], canal])
  x_s= Conv2D(canal*2, 1, padding= 'same')(out)

  return x_s

def split_heads(x, batch_size, heads, depth):
  x= tf.reshape(x, (batch_size, -1, heads, depth))
  return tf.transpose(x, perm=[0, 2, 1, 3])

def scaled_dot_product_attention(q, v, k):
  matmul_qk= tf.matmul(q, k, transpose_b= True)
  dk= tf.cast(tf.shape(k)[-1], tf.float32)
  scaled_attention_logist= matmul_qk/ tf.math.sqrt(dk)

  attention_w= Activation('softmax')(scaled_attention_logist)
  return tf.matmul(attention_w, v)

def conv_code(entrada, filtros, ker_reg= None, decoNormal= False):
  x= Conv2D(filtros, 3, padding= "same", activation= "relu", kernel_regularizer= ker_reg)(entrada)
  x= BatchNormalization()(x)
  x= Conv2D(filtros, 3, padding= "same", activation= "relu", kernel_regularizer= ker_reg)(x)
  x= BatchNormalization()(x)
  x= Conv2D(filtros, 3, padding= "same", activation= "relu", kernel_regularizer= ker_reg)(x)
  x= BatchNormalization()(x)

  if False == decoNormal:
    x1 = MaxPooling2D(3, strides= 2, padding= "same")(x)
    return x1, x 
  if 64== filtros and True == decoNormal:
    x1 = MaxPooling2D(3, strides= 2, padding= "same")(x)
    x2 = MaxPooling2D(3, strides= 4, padding= "same")(x)
    x3 = MaxPooling2D(3, strides= 8, padding= "same")(x)
    return x, x1, x2, x3
  if 128 == filtros and True == decoNormal:
    x1 = MaxPooling2D(3, strides= 2, padding= "same")(x)
    x2 = MaxPooling2D(3, strides= 4, padding= "same")(x)
    return x, x1, x2
  if 256 == filtros and True == decoNormal:
    x1 = MaxPooling2D(3, strides= 2, padding= "same")(x)
    return x, x1
  if 512 == filtros and True == decoNormal:
    return x, MaxPooling2D(3, strides= 2, padding= "same")(x)

def attentionModule(filtros, y= [], Normal= False):
  if 64 == filtros:
    return second_p(y, [10, 5, 5, 5, 2, 5], [140, 120, 80, 50, 20, 10], 160, 64)
  if 128 == filtros:
    z = Concatenate()([y[0], y[1]]) if True== Normal else y
    return second_p(z, [7, 6, 5, 3, 2, 5], [70, 60, 50, 30, 20, 10], 80, 128)
  elif 256 == filtros:
    if True == Normal:
      z = Concatenate()([y[0], y[1]])
      z = Concatenate()([z, y[2]])
    else:
      z= y
    return second_p(z, [3, 2, 2, 5], [30, 16, 12, 10], 40, 256)
  elif 512 == filtros:
    if True == Normal:
      z = Concatenate()([y[0], y[1]])
      z = Concatenate()([z, y[2]])
      z = Concatenate()([z, y[3]])
    else:
      z= y
    return second_p(z, [5], [10], 20, 512)

def dconv_code(entrada, filtros, ker_reg= None):
  x= Conv2D(filtros, 3, padding= "same", activation= "relu", kernel_regularizer= ker_reg, dilation_rate= 3)(entrada)
  x= BatchNormalization()(x)
  x= Conv2D(filtros, 3, padding= "same", activation= "relu", kernel_regularizer= ker_reg, dilation_rate= 3)(x)
  x= BatchNormalization()(x)
  x= Conv2D(filtros, 3, padding= "same", activation= "relu", kernel_regularizer= ker_reg, dilation_rate= 3)(x)
  x= BatchNormalization()(x)
  residual= x
  x= MaxPooling2D(3, strides= 2, padding= "same")(x)
  return residual, x

def botle(entrada, filtros= 1024, ker_reg= None, typec = 'Normal'):
  x= Conv2D(filtros, 3, padding= "same", activation= "relu",  kernel_regularizer= ker_reg)(entrada)
  x= BatchNormalization()(x)
  x= Conv2D(filtros, 3, padding= "same", activation= "relu",  kernel_regularizer= ker_reg)(x)
  x= BatchNormalization()(x)

  x= Conv2DTranspose(filtros, 3,padding= "same" )(x)

  if typec != 'Normal':
    y= UpSampling2D(4)(x)
    y1 = UpSampling2D(8)(x)
    y2 = UpSampling2D(16)(x)
    return x, y, y1, y2
  else:
    return x

def conv_decode(entrada, filtros, ker_reg= None, typec= 'Normal'):
  x= Conv2D(filtros, 3, padding= "same", activation= "relu", kernel_regularizer= ker_reg)(entrada)
  x= BatchNormalization()(x)
  x= Conv2D(filtros, 3, padding= "same", activation= "relu", kernel_regularizer= ker_reg)(x)
  x= BatchNormalization()(x)
  x= Conv2D(filtros, 3, padding= "same", activation= "relu", kernel_regularizer= ker_reg)(x)
  x= BatchNormalization()(x)

  x= Conv2DTranspose(filtros//2, 3,padding= "same" )(x)

  if  512== filtros and typec != 'Normal':
    z = UpSampling2D(4)(x)
    y = UpSampling2D(8)(x)
    return x, z, y
  
  if 256== filtros and typec != 'Normal':
    z = UpSampling2D(4)(x)
    return x, z
  
  if 128 == filtros or typec == 'Normal': return x

def crop_and_cat(filtros, res, z, valType= True):
  if True == valType:
    x= UpSampling2D(2)(res)
    return Concatenate()([x, z])

  if 1024 == filtros and False == valType:
    x= UpSampling2D(2)(res)
    return Concatenate()([x, z])

  if 512 == filtros and False == valType:
    x= UpSampling2D(2)(res)
    y= Concatenate()([x, z[0]])

    return Concatenate()([y, z[1]])

  if 256 == filtros and False == valType:
    x= UpSampling2D(2)(res)
    y= Concatenate()([x, z[0]])
    y1= Concatenate()([y, z[1]])

    return Concatenate()([y1, z[2]])

  if 128 == filtros and False == valType:
    x= UpSampling2D(2)(res)
    y= Concatenate()([x, z[0]])
    y1= Concatenate()([y, z[1]])
    y2= Concatenate()([y1, z[2]])

    return Concatenate()([y2, z[3]])
  
def capa_final(entrada, filtros, num_clases):
  x= Conv2D(filtros, 3, padding= "same", activation= "relu")(entrada)
  x= BatchNormalization()(x)
  x= Conv2D(filtros, 3, padding= "same", activation= "relu")(x)
  x= BatchNormalization()(x)
  x= Conv2D(filtros, 3, padding= "same", activation= "relu")(x)
  x= BatchNormalization()(x)

  x= Conv2D(num_clases, 3, padding= "same", activation= "softmax")(x)

  return x

def get_model(img_size, num_classes, valType = True):
  inputs= tf.keras.Input(shape= img_size)
  conv_64, res64 = conv_code(inputs, 64)
  conv_128, res128 = conv_code(conv_64, 128)
  conv_256, res256= conv_code(conv_128, 256)
  conv_512, res512 = conv_code(conv_256, 512)
  res128 = attentionModule(128, res128)
  res256 = attentionModule(256, res256)
  res512 = attentionModule(512, res512)
  bot = botle(conv_512, typec= 'Normal')
  crop1= crop_and_cat(1024, bot, res512)
  decod_3 = conv_decode(crop1, 512, typec= 'Normal')
  crop2 = crop_and_cat(512, decod_3, res256)
  decod_2 = conv_decode(crop2, 256, typec= 'Normal')
  crop3 = crop_and_cat(256, decod_2, res128)
  decod_1 = conv_decode(crop3, 128, typec= 'Normal')
  crop4 = crop_and_cat(128, decod_1, res64)
  outputs= capa_final(crop4, 64, num_classes)
  model= Entre_editado(inputs= inputs, outputs= outputs)

  return model