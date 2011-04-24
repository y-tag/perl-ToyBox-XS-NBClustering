#!/usr/bin/perl

use strict;
use warnings;

use Data::Dumper;

use lib qw(lib blib/lib blib/arch);
use ToyBox::XS::NBClustering;

my $nb = ToyBox::XS::NBClustering->new();

$nb->add_instance(attributes => {a => 2, b => 3});

my $attributes = {c => 1, d => 4};
$nb->add_instance(attributes => $attributes);

$nb->train(cluster_num => 2, seed => 1000, alpha => 1.0);

my $result = $nb->predict(attributes => {a => 2, b => 3});
print Dumper($result);

$attributes = {a => 1, b => 1, c => 1, d => 1};
$result = $nb->predict(attributes => $attributes);
print Dumper($result);
