program example_mnist

  ! A training example with the MNIST dataset.
  ! Uses stochastic gradient descent and mini-batch size of 100.
  ! Can be run in serial or parallel mode without modifications.

  use mod_kinds, only: ik, rk
  use mod_mnist, only: label_digits, load_mnist
  use mod_network, only: network_type

  implicit none

  real(rk), allocatable :: tr_images(:,:), tr_labels(:)
  real(rk), allocatable :: te_images(:,:), te_labels(:)
  !real(rk), allocatable :: va_images(:,:), va_labels(:)
  real(rk), allocatable :: input(:,:), output(:,:)

  type(network_type) :: net

  integer(ik) :: i, n, num_epochs, im
  integer(ik) :: batch_size, batch_start, batch_end
  real(rk) :: pos

  call load_mnist(tr_images, tr_labels, te_images, te_labels)

  ! net = network_type([784, 10, 10])
  ! net = network_type([784, 30, 10])
  call net % load('../../src/my_net.txt')
  write(*,*) 'loaded ../../src/my_net.txt'

  batch_size = 1000
  num_epochs = 10

  ! if (this_image() == 1) then
  !    ! write(*, '(a,f5.2,a)') 'Initial accuracy: ',&
  !    !   net % accuracy(te_images, label_digits(te_labels)) * 100, ' %'
  !   write(*,*) 'inside if'
  ! end if

  epochs: do n = 1, num_epochs
    write(*,*) 'epochs?'
    mini_batches: do i = 1, size(tr_labels) / batch_size

      ! pull a random mini-batch from the dataset
      call random_number(pos)
      batch_start = int(pos * (size(tr_labels) - batch_size + 1))
      batch_end = batch_start + batch_size - 1

      ! prepare mini-batch
      write(*,*) 'tr_labels?'
      input = tr_images(:,batch_start:batch_end)
      write(*,*) 'label digits?'
      output = label_digits(tr_labels(batch_start:batch_end))
      ! call net % fwdprop(input(:,1))
      im = size(input, dim=2)
      write(*,*) im
      im = size(input,dim=1)
      write(*,*) im

      net%layers(1) % a = input(:,1)
      net%layers(2) % z = matmul(transpose(net%layers(1) % w), net%layers(1) % a) + net%layers(2) % b
      net%layers(2) % a = net%layers(2) % activation(net%layers(2) % z)

      net%layers(3) % z = matmul(transpose(net%layers(2) % w), net%layers(2) % a) + net%layers(3) % b
      net%layers(3) % a = net%layers(3) % activation(net%layers(3) % z)
      ! train the network on the mini-batch
      write(*,*) 'train?'
      call net % train(input, output, eta=3._rk)

    end do mini_batches
    write(*,*) 'after mini batches'
    if (this_image() == 1) then
      write(*,*) 'attempting accuracy again'
      write(*, '(a,i2,a,f5.2,a)') 'Epoch ', n, ' done, Accuracy: ',&
        net % accuracy(te_images, label_digits(te_labels)) * 100, ' %'
    end if

  end do epochs
  call net % save('../../src/my_net.txt')
end program example_mnist
