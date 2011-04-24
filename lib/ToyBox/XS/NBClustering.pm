package ToyBox::XS::NBClustering;

use 5.0080;
use strict;
use warnings;

require Exporter;

our $VERSION = '0.01';

require XSLoader;
XSLoader::load('ToyBox::XS::NBClustering', $VERSION);

sub add_instance {
    my ($self, %params) = @_;

    die "No params: attributes" unless defined($params{attributes});
    my $attributes = $params{attributes};
    my $label      = $params{label};
    die "attributes is not hash ref"   unless ref($attributes) eq 'HASH';
    die "attributes is empty hash ref" unless keys %$attributes;

    my %copy_attr = %$attributes;

    if (defined($label)) {
        $self->xs_add_labeled_instance(\%copy_attr, $label);
    } else {
        $self->xs_add_instance(\%copy_attr);
    }
    1;
}

sub train{
    my ($self, %params) = @_;

    my $cluster_num = $params{cluster_num};
    $cluster_num = 2 unless defined($cluster_num);
    die "cluster_num is le 1" unless $cluster_num > 1;

    my $max_iteration = $params{max_iteration} || 100;
    my $epsilon = $params{epsilon} || 1e-10;
    my $alpha = $params{alpha} || 1.0;
    my $seed = $params{seed} || int(rand(2 ** 32));

    $self->xs_train($cluster_num, $max_iteration, $epsilon, $alpha, $seed);
    1;
}

sub predict {
    my ($self, %params) = @_;

    die "No params: attributes" unless defined($params{attributes});
    my $attributes = $params{attributes};
    die "attributes is not hash ref"   unless ref($attributes) eq 'HASH';
    die "attributes is empty hash ref" unless keys %$attributes;

    my $result = $self->xs_predict($attributes);

    $result;
}


1;
__END__
=head1 NAME

ToyBox::XS::NBClustering - Simple Naive Bayes Clustering using Perl XS

=head1 SYNOPSIS

  use ToyBox::XS::NBClustering;

  my $nb = ToyBox::XS::NBClustering->new();
  
  $nb->add_instance(
      attributes => {a => 2, b => 3},
  );
  
  $nb->add_instance(
      attributes => {c => 3, d => 1},
  );
  
  $nb->train(cluster_num => 2, alpha => 1.0);
  
  my $probs = $nb->predict(
                  attributes => {a => 1, b => 1, d => 1, e =>1}
              );

=head1 DESCRIPTION

This module implements a simple Naive Bayes Clustering using Perl XS.

=head1 AUTHOR

TAGAMI Yukihiro <tagami.yukihiro@gmail.com>

=head1 LICENSE

This library is distributed under the term of the MIT license.

L<http://opensource.org/licenses/mit-license.php>
