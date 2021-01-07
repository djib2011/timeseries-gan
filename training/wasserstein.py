import tensorflow as tf


def gradient_penalty(real_data, generated_data):

    batch_size = real_data.shape[0]

    # Calculate interpolation
    alpha = tf.random.uniform((batch_size,))
    alpha = tf.reshape(alpha, real_data.shape)


    interpolated = alpha * real_data + (1 - alpha) * generated_data
    interpolated = Variable(interpolated, requires_grad=True)
    if self.use_cuda:
        interpolated = interpolated.cuda()

    # Pass interpolated data through Critic
    prob_interpolated = self.c(interpolated)

    # Calculate gradients of probabilities with respect to examples
    gradients = torch_grad(outputs=prob_interpolated, inputs=interpolated,
                           grad_outputs=torch.ones(prob_interpolated.size()).cuda() if self.use_cuda
                           else torch.ones(prob_interpolated.size()), create_graph=True,
                           retain_graph=True)[0]
    # Gradients have shape (batch_size, num_channels, series length),
    # here we flatten to take the norm per example for every batch
    gradients = gradients.view(batch_size, -1)
    self.losses['gradient_norm'].append(gradients.norm(2, dim=1).mean().data.item())

    # Derivatives of the gradient close to 0 can cause problems because of the
    # square root, so manually calculate norm and add epsilon
    gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)

    # Return gradient penalty
    return self.gp_weight * ((gradients_norm - 1) ** 2).mean()
