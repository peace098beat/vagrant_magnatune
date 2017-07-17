# -*- mode: ruby -*-
# vi: set ft=ruby :

Vagrant.configure('2') do |config|
  config.vm.box      = 'bento/ubuntu-16.04'
  config.vm.hostname = 'deep-learning'

  # config.vm.network :forwarded_port, guest: 3000, host: 3000
  # config.vm.network :forwarded_port, guest: 8888, host: 8888

  config.vm.synced_folder "data/", "/home/vagrant/data"

  config.vm.provider "virtualbox" do |v|
    v.memory = 1024
    v.cpus = 4
  end
  
  config.vm.provision :shell, path: 'bootstrap.sh', keep_color: true
  
end