using FluentValidation;

namespace Derma.Application.CosmeticUsage
{
    public class CosmeticUsageValidator : AbstractValidator<Derma.Domain.CosmeticUsage>
    {
        public CosmeticUsageValidator()
        {
            RuleFor(x => x.productName)
                .NotEmpty().WithMessage("Nazwa produktu jest wymagana.")
                .MaximumLength(200).WithMessage("Nazwa produktu może mieć maksymalnie 200 znaków.");

            RuleFor(x => x.category)
                .IsInEnum().WithMessage("Nieprawidłowa kategoria.");

            RuleFor(x => x.startDate)
                .NotEmpty().WithMessage("Data rozpoczęcia jest wymagana.")
                .Must(d => d <= DateTime.UtcNow.AddDays(1))
                .WithMessage("Data rozpoczęcia nie może być z przyszłości.");

            RuleFor(x => x.endDate)
                .Must((x, end) => end == null || end >= x.startDate)
                .WithMessage("Data zakończenia nie może być wcześniejsza niż data rozpoczęcia.");

            RuleFor(x => x.frequency)
                .MaximumLength(200).WithMessage("Częstotliwość może mieć maksymalnie 200 znaków.");

            RuleFor(x => x.notes)
                .MaximumLength(1000).WithMessage("Notatka może mieć maksymalnie 1000 znaków.");
        }
    }
}