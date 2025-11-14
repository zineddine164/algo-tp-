from django import forms

class MatrixForm(forms.Form):
    matrix = forms.CharField(
        widget=forms.Textarea(attrs={
            'rows': 8,
            'placeholder': 'Entrez la matrice: une ligne = une rangée. Séparez les valeurs par des espaces ou des virgules.'
        }),
        label='Matrice',
        help_text='Ex:\n1 2 3\n4 5 6\n7 8 9'
    )

    remove_duplicates = forms.BooleanField(required=False, initial=False, label='Supprimer doublons avant insertion')
