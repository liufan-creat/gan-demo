

#' Title Fits the model on data yielded batch-by-batch by triplet-cGAN
#'
#' @param generator A generator (e.g. like the one provided by flow_images_from_directory() or a custom R generator function).
#' The output of the generator must be a list of one of these forms: <br> - (inputs, targets) <br> - (inputs, targets, sample_weights) <br>.
#'  This list (a single output of the generator) makes a single batch.
#'  Therefore, all arrays in this list must have the same length (equal to the size of this batch).
#'  Different batches may have different sizes. For example, the last batch of the epoch is commonly smaller than the others, if the size of the dataset is not divisible by the batch size.
#'  The generator is expected to loop over its data indefinitely. An epoch finishes when steps_per_epoch batches have been seen by the model.
#' @param discriminator network for authenticity
#' @param n.discriminator Integer.Number of cycle for discriminator in each epoch
#' @param batch Integer.Number of samples in each training
#' @param epoch Integer. Number of epochs to train the model
#' An epoch is an iteration over the entire data provided,Note that in conjunction with initial_epoch,epochs is to be understood as “final epoch”.
#' The model is not trained for a number of iterations given by epochs,  but merely until the epoch of index epochs is reached.
#' @param learn_rate 	float >= 0. Learning rate.
#' @param object RNAseq matrix
#' @param ture_lab group label of RNAseq
#' @param classification network for classification
#' @param n.label length of ture_lab
#' @param distance_metric str or a Callable that determines distance metric. Valid strings are "L2" for l2-norm distance,
#' "squared-L2" for squared l2-norm distance, and "angular" for cosine similarity.
#' @param margin Float, margin term in the loss definition. Default value is 1.0.
#'
#' @return epoch, gloss,dloss, pearson correlations
#' @export
#'
#' @examples
#' \dontrun{
#' library(keras)
#' library(reticulate)
#' library(tensorflow)
#' library(bulkGAN)
#' library(Hmisc)
#' tfa <- import('tensorflow_addons')  #   import tensorflow_addons model
#'
#' fit_SNGAN(generator, discriminator,n.discriminator=5,batch=25,
#'            epoch = 500,learn_rate = 0.0001,object = normal_tissure )
#'
#'}


fit_triplet_cgan <- function(generator, discriminator,n.discriminator=5,batch=25,
                      epoch = 500,learn_rate = 0.0001,classification,n.label=3,
                      object = NULL,ture_lab=NULL ,
                      distance_metric = "squared-L2",margin = 0.1){


  wloss <- function(y_true,y_pred){
    loss <- -1 *  keras::k_mean(y_true * y_pred)
    return(loss)
  }

  l <- 100

  dloss <- NULL
  gloss <- NULL
  closs <- NULL

  g_coor_p <- NULL

  discriminator %>% keras::compile(optimizer = keras::optimizer_rmsprop(learning_rate = learn_rate),
                            loss = wloss )

  freeze_weights(discriminator)

  input <- layer_input(shape = l)
  gan <-  input %>% generator %>%discriminator

  GAN <- keras::keras_model(input,gan)
  GAN %>% keras::compile(optimizer = keras::optimizer_rmsprop(learning_rate = learn_rate ),
                  loss = wloss)


  classification %>% keras::compile(optimizer = keras::optimizer_rmsprop(learning_rate = learn_rate)  )
  generator %>% keras::compile(optimizer = keras::optimizer_rmsprop(learning_rate = learn_rate)  )


  for (i in 1:epoch) {

    unfreeze_weights(discriminator)
    noise_label <- sample(0:n.label,batch,replace = T) %>% matrix(ncol = 1)


    for (j in 1:n.discriminator) {
      id <- sample(colnames(object),batch,replace = F)
      real <- object[,id]
      real1 <- t(real)

      id <- sample(colnames(object),batch,replace = F)
      real <- object[,id]
      real2 <- t(real)

      real_res <- array(runif(batch,-1, -1),dim =c(batch,1))
      d_loss_real <- discriminator%>%keras::train_on_batch(list(real1,real2),real_res)

      noise <- keras::k_random_normal(c(batch,l),mean = 0,stddev = 1)
      fake <- keras::predict_on_batch(generator,noise)
      fake_res <- array(runif(batch,1,1),dim =c(batch,1))
      d_loss_fake <-discriminator%>%keras::train_on_batch(list(real1,fake),fake_res)

      dloss[i] <- 0.5*( d_loss_fake+d_loss_real)

    }

    gc()
    freeze_weights(discriminator)
    noise <- keras::k_random_normal(c(batch,l),mean = 0,stddev = 1)
    real_res <- array(runif(batch, -1 ,  -1),dim =c(batch,1))
    gloss[i]  <- GAN%>%keras::train_on_batch(list(noise,noise_label,real1),real_res)


    with(tf$GradientTape()%as% tape, {
      fake <- generator(list(noise,noise_label),training = F)
      matrix_log <- classification(rbind(t(object),fake),training = T)

      pre_lab <- rbind( ture_lab, array(runif(batch,n.label+1,n.label+1),dim =c(batch,1)) )
      c_loss<-tfa$losses$triplet_hard_loss(pre_lab,matrix_log,
                                           distance_metric = "squared-L2",margin = margin)

      c_grad <- tape$gradient(c_loss,classification$trainable_variables)
      classification$optimizer$apply_gradients(zip_lists(c_grad,classification$trainable_variables))
    })
    closs[i]  <- as.array(c_loss)

    with(tf$GradientTape()%as% tape, {
      fake <- generator(list(noise,noise_label),training = T)
      matrix_log <- classification(rbind(t(object),fake),training = F)

      pre_lab <- rbind(ture_lab, noise_label)
      #g_loss <- tfa$losses$triplet_semihard_loss(pre_lab,rbind(real_log,fake_log), distance_metric = "squared-L2",margin = 0.1)
      cg_loss<-tfa$losses$triplet_hard_loss(pre_lab,matrix_log,
                                            distance_metric = "squared-L2",margin = margin)

      cg_grad <- tape$gradient(cg_loss,generator$trainable_variables)
      generator$optimizer$apply_gradients(zip_lists(cg_grad,generator$trainable_variables))
    })



    fake <- keras::predict_on_batch(generator,noise)
    data_t <- t(rbind(fake,real))
    x_corr <- Hmisc::rcorr(data_t,type = 'pearson')
    r <- x_corr$r
    r <- as.data.frame(r)[1:batch,batch+1:2*batch]
    g_coor_p[i] <- mean(as.matrix(r))


    print(i)
    print(  dloss[i])
    print(  gloss[i])
    print( g_coor_p[i] )
  }


}
